import ffmpeg

media_path = '/ssd1/DF/FakeAVCeleb_v1.2/video/FakeVideo-FakeAudio/Caucasian (European)/men/id00305/00113_1_id00185_wavtolip.mp4'
probe = ffmpeg.probe(media_path)

print(probe)