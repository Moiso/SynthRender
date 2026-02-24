import subprocess

def run_bproc_cli(command:list[str]):
    ret = False
    try:
        print(f"Command: [{' '.join(command)}]")
        process = subprocess.Popen(command)
        process.wait() # Wait for the process to complete or be interrupted

        ret = True

    except subprocess.CalledProcessError as e:
        print(f"Error while running BlenderProc command: {e}")
    except KeyboardInterrupt:
        print("Process interrupted. Terminating annotator...")
        process.terminate()
        process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()

        return ret
