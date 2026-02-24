import time
import subprocess
import argparse
import os
import time 
import threading

from tqdm.auto import tqdm


from synthrender.utils import misc_utils

def parse_args() -> str:
    parser = argparse.ArgumentParser(description="Process arguments.")

    parser.add_argument('--num-frames', '-n', type=int, help="Number of frames to render", default=-1)
    parser.add_argument("--num-gpus", '-g', type=int, help="Number of GPUs to use, if not specified it will use them all", default=-1)
    parser.add_argument("--config", '-c', type=str, help="Path to config file", default="config.yaml")

    parser.add_argument("--start-frame", '-s', type=int, help="Frame at which the render should start (inclusive). If set, stop-frame must be set too.", default=-1)
    parser.add_argument("--stop-frame", '-e', type=int, help="Frame at which the render should stop (inclusive). If set, start-frame will be set to 0 unless it has already being set.", default=-1)

    args = parser.parse_args()

    if args.num_frames == -1:
        parser.print_help()
        exit()

    # If start-frame is set but stop-frame not, raise issue.
    if args.start_frame != -1 and args.stop_frame == -1:
        parser.error("--stop-frame is required when --start-frame is specified.")

    # If stop-frame is set but start-frame not, raise issue.
    if args.stop_frame != -1 and args.start_frame == -1:
        parser.error("--start-frame is required when --stop-frame is specified.")

    if args.stop_frame >= args.num_frames:
        parser.error("--end-frame cannot be bigger than --num-frames -1 (Keyframing goes from 0 to n-1)")
    

    return args.config, args.num_frames, args.num_gpus, args.start_frame, args.stop_frame

def update_pbars_with_stdout(config_path: str, batches: list, processes: list[subprocess.Popen]):
    # Load configuration and extract sample count
    if not os.path.exists(config_path):
        raise Exception("Error: No config file found!")
    
    config = misc_utils.load_config(config_path)
    n_samples = config["cycles_samples"]
    
    fmt = '{percentage:3.0f}%[{bar:33}] {n_fmt:>5}/{total_fmt:<5} [{elapsed}<{remaining:>8} - {rate_fmt:>14}] {desc}'
    keyargs_parent = {"desc": "-:-", "ascii": " =", "leave": True, "unit": "frame"}
    keyargs_child  = {"desc": "-:-", "ascii": " -", "leave": True, "unit": "sample", "total": n_samples}
    YELLOW = '\033[93m'
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"

    try:
        pbars_parent = [tqdm(**keyargs_parent, total=batches[i][1]-batches[i][0]+1, position=i*3,
                            bar_format=f"GPU_{i} frames rendered {fmt}") for i in range(len(processes))]
        pbars_childs = [tqdm(**keyargs_child, position=i*3+1,
                            bar_format=f"GPU_{i} frame samples   {fmt}") for i in range(len(processes))]
        
        def process_line(idx: int, line: str):
            if line.startswith("Time: "):
                if pbars_parent[idx].n < pbars_parent[idx].total:
                    pbars_parent[idx].n+=1
            elif line.startswith("Fra:"):
                cols = [col.strip() for col in line.split("|")]
                if 'Scene, ViewLayer' in cols:
                    status = " | ".join(cols[cols.index("Scene, ViewLayer")+1:])
                    pbars_childs[idx].desc = status
                    pbars_parent[idx].desc = f"Rendering frame {batches[idx][0] + pbars_parent[idx].n}"
                    if status.startswith('Sample'):
                        try:
                            pbars_childs[idx].n = int(status[len('Sample'):].split('/', maxsplit=1)[0])
                        except ValueError:
                            pass
                if 'Composting' in cols:
                    pbars_childs[idx].desc = " | ".join(cols[cols.index("Composting")+1:])
            elif line.startswith("[update]"):
                pbars_parent[idx].desc = line.strip("[update] ").strip()
            elif line.lower().startswith(("warning", "error", "oserror")):
                tqdm.write(f"{YELLOW}On GPU [{idx}]: {line.strip()}{RESET}\n")

            pbars_parent[idx].refresh()
            pbars_childs[idx].refresh()

        def reader_thread(process: subprocess.Popen, idx: int):
            pbar = pbars_parent[idx]
            cbar = pbars_childs[idx]

            while (line := process.stdout.readline()):
                process_line(idx, line)

            process.stdout.close()

            if pbar.n == pbar.total:
                pbar.colour, cbar.colour = "green", "green"
                pbar.desc, cbar.desc = f"{GREEN}Render completed{RESET}", f"{GREEN}Finished{RESET}"
            else:
                pbar.colour, cbar.colour = "red", "red"
                pbar.desc, cbar.desc = f"{RED}Render not completed{RESET}", F"{RED}Finished{RESET}"

            parent_string = pbar.format_meter(**pbar.format_dict)
            child_string  = cbar.format_meter(**cbar.format_dict)
            pbar.bar_format = parent_string 
            cbar.bar_format = child_string 

            pbar.refresh()
            cbar.refresh()
            
            # Checking for errors to print:
            errors = process.stderr.read()
            process.stderr.close()

            if errors:
                tqdm.write(f"{RED}On GPU [{idx}]:{RESET}")
                tqdm.write(f"{RED}{errors}{RESET}")
                tqdm.write("")

        threads = [threading.Thread(target=reader_thread, args=(proc, idx)) 
                for idx, proc in enumerate(processes)]
        
        for t in threads: t.start()
        for t in threads: t.join()

    finally:
        # Terminate any still-running subprocesses
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
                
        # Refresh and close progress bars
        for pbar, cbar in zip(pbars_parent, pbars_childs):
            pbar.close(); cbar.close()
            print()

def run_parallel_create_syn_data(mode:str, script_path:str, config_path:str, keyframes:int, n_gpus:int=0, start_frame=0, stop_frame=-1):
    ret = False

    try:
        processes:list[subprocess.Popen] = []
        batches:list[tuple] = []

        n_gpus = -1 if n_gpus < 1 else n_gpus # Sets to -1 if 0 gpus are set.
        gpu_ids = list(range(n_gpus)) if n_gpus != -1 else [-1] # List with ids [0,1,2,..,n] or id -1 if no gpu specified.
        start_frame = 0 if start_frame == -1 else start_frame
        stop_frame = keyframes-1 if stop_frame == -1 else stop_frame

        intervals = misc_utils.split_interval(start_frame, stop_frame, len(gpu_ids))

        for gpu_id, interval in zip(gpu_ids, intervals):
            start, stop = interval

            command = ["blenderproc", mode, script_path, "--verbose", "-n", str(keyframes), "-s", str(start), "-e", str(stop), "-g", str(gpu_id), "-c", config_path]
            print(f"Command: [{' '.join(command)}]")

            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(process)
            batches.append((start, stop))

        # Blocking function that updates progress bars
        print()
        update_pbars_with_stdout(config_path, batches, processes)

        ret = True


    except subprocess.CalledProcessError as e:
        print(f"Error while running BlenderProc command: {e}")
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.wait()      # Ensure the subprocess is completely terminated
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        
    return ret


if __name__ == "__main__":
    config_file_path, n_frames, n_gpus, start_frame, stop_frame = parse_args()

    from synthrender.scripts import __path__ as scripts_path
    blenderproc_script = os.path.join(scripts_path[0], 'create_synthetic_data.py')

    assert os.path.exists(config_file_path), f"Error: Could not find config file in: {config_file_path}"
    assert os.path.exists(blenderproc_script), f"Error: Could not find script file in: {blenderproc_script}"


    start = time.time()
    ret = run_parallel_create_syn_data("run", blenderproc_script, config_file_path, keyframes=n_frames, n_gpus=n_gpus, start_frame=start_frame, stop_frame=stop_frame)
    render_elapsed = time.time()-start

    print(f"\n> Total rendering time: {render_elapsed:.3f} s")