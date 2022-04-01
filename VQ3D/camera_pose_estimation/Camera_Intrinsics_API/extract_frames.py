import os
import av

class FrameExtractor():
    def extract(self, filename: str, output_dir: str) -> None:
        reader = av.open(filename, 'r')
        for i, frame in enumerate(reader.decode(video=0)):
            image = frame.to_image()
            image.save(os.path.join(output_dir,
                                    "color_{:07d}.jpg".format(i)
                                   )
                      )

