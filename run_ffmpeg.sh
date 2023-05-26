# extract
yes | ffmpeg -ss 00:00:00 -i test1_line.mp4 -t 00:00:10 -vf fps=30 test1_line_short.mp4

# export mp4
# ffmpeg -i runs/track/test1_line/test1.mp4 -vf fps=30 -vcodec libx264 test1_line.mp4

# ffmpeg -i test_data/vehicle-count.mp4 -vf fps=30 -vcodec libx264 vehicle-count.mp4
