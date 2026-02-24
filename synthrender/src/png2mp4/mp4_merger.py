import ffmpeg

class mp4Merger:

    @staticmethod
    def create_mp4_default(input_sequence:str='rgb_*.png', output:str="animation/", framerate:int=24):
        # 1. Declare your input sequence (and any per‑input options)
        stream = ffmpeg.input(
            input_sequence,   # pattern for your image files: RGB_0001.png, RGB_0002.png, …
            framerate=framerate,     # “-framerate 24”
            pattern_type='glob'
        )

        # 2. Chain an output, along with all the codec/container options
        stream = stream.output(
            output,  # final file path
            vcodec='libx264', # “-c:v libx264”
            crf=18,           # “-crf 18”  (quality level: lower = better)
            pix_fmt='yuv420p' # “-pix_fmt yuv420p” for compatibility
        )

        # 3. (Optional) Add any global args, e.g. to hide the banner
        stream = stream.global_args('-hide_banner')

        # 4. Execute the process
        stream.run()