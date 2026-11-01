# Recording a demo video

Records a live Spark Talk episode in 4K, then renders a 1080p, 30 fps video with smooth camera moves
(zoom in, pan, zoom out) to whatever is happening: the topic being typed, the search, the sources, the hosts,
the under-the-hood views. There's no text on the video, so the dashboard speaks for itself.

```sh
sudo tools/recording/start-gpu-display.sh          # once: a 4K display on the GPU (display :1)
python3 tools/recording/record.py "Your topic" tools/recording/takes/t1 1
python3 tools/recording/render.py tools/recording/takes/t1
sudo systemctl stop rec-xorg                       # when you're done
```

The video lands in `takes/t1/spark-talk.mp4`, at -14 LUFS (the usual loudness for social video).
The shot list lives in `record.py`: each `shot(...)` call names what the camera should frame and when.
