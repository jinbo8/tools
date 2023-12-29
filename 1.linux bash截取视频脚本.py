
"""

# 1.使用linux bash 将长视频切成短视频

videoname="45minR1_Dn_CamE_cartruck.mp4"
cam="45minR1_Dn_CamE_cartruck"
for n in 1
do
    ffmpeg -i $videoname -c copy -ss 00:00:00 -to 00:01:00 ${cam}1.mp4
    ffmpeg -i $videoname -c copy -ss 00:01:00 -to 00:02:00 ${cam}2.mp4
    ffmpeg -i $videoname -c copy -ss 00:02:00 -to 00:03:00 ${cam}3.mp4
    ffmpeg -i $videoname -c copy -ss 00:03:00 -to 00:04:00 ${cam}4.mp4
    ffmpeg -i $videoname -c copy -ss 00:04:00 -to 00:05:00 ${cam}5.mp4
    ffmpeg -i $videoname -c copy -ss 00:05:00 -to 00:06:00 ${cam}6.mp4
    ffmpeg -i $videoname -c copy -ss 00:06:00 -to 00:07:00 ${cam}7.mp4
    ffmpeg -i $videoname -c copy -ss 00:07:00 -to 00:08:00 ${cam}8.mp4
    ffmpeg -i $videoname -c copy -ss 00:08:00 -to 00:09:00 ${cam}9.mp4
    ffmpeg -i $videoname -c copy -ss 00:09:00 -to 00:10:00 ${cam}10.mp4
    ffmpeg -i $videoname -c copy -ss 00:10:00 -to 00:11:00 ${cam}11.mp4
    ffmpeg -i $videoname -c copy -ss 00:11:00 -to 00:12:00 ${cam}12.mp4
    ffmpeg -i $videoname -c copy -ss 00:12:00 -to 00:13:00 ${cam}13.mp4
    ffmpeg -i $videoname -c copy -ss 00:13:00 -to 00:14:00 ${cam}14.mp4
    ffmpeg -i $videoname -c copy -ss 00:14:00 -to 00:15:00 ${cam}15.mp4
    ffmpeg -i $videoname -c copy -ss 00:15:00 -to 00:16:00 ${cam}16.mp4
    ffmpeg -i $videoname -c copy -ss 00:16:00 -to 00:17:00 ${cam}17.mp4
    ffmpeg -i $videoname -c copy -ss 00:17:00 -to 00:18:00 ${cam}18.mp4
    ffmpeg -i $videoname -c copy -ss 00:18:00 -to 00:19:00 ${cam}19.mp4
    ffmpeg -i $videoname -c copy -ss 00:19:00 -to 00:20:00 ${cam}20.mp4
    ffmpeg -i $videoname -c copy -ss 00:20:00 -to 00:21:00 ${cam}21.mp4
    ffmpeg -i $videoname -c copy -ss 00:21:00 -to 00:22:00 ${cam}22.mp4
    ffmpeg -i $videoname -c copy -ss 00:22:00 -to 00:23:00 ${cam}23.mp4
    ffmpeg -i $videoname -c copy -ss 00:23:00 -to 00:24:00 ${cam}24.mp4
    ffmpeg -i $videoname -c copy -ss 00:24:00 -to 00:25:00 ${cam}25.mp4
    ffmpeg -i $videoname -c copy -ss 00:25:00 -to 00:26:00 ${cam}26.mp4
    ffmpeg -i $videoname -c copy -ss 00:26:00 -to 00:27:00 ${cam}27.mp4
    ffmpeg -i $videoname -c copy -ss 00:27:00 -to 00:28:00 ${cam}28.mp4
    ffmpeg -i $videoname -c copy -ss 00:28:00 -to 00:29:00 ${cam}29.mp4
    ffmpeg -i $videoname -c copy -ss 00:29:00 -to 00:30:00 ${cam}30.mp4
    ffmpeg -i $videoname -c copy -ss 00:30:00 -to 00:31:00 ${cam}31.mp4
    ffmpeg -i $videoname -c copy -ss 00:31:00 -to 00:32:00 ${cam}32.mp4
    ffmpeg -i $videoname -c copy -ss 00:32:00 -to 00:33:00 ${cam}33.mp4
    ffmpeg -i $videoname -c copy -ss 00:33:00 -to 00:34:00 ${cam}34.mp4
    ffmpeg -i $videoname -c copy -ss 00:34:00 -to 00:35:00 ${cam}35.mp4
    ffmpeg -i $videoname -c copy -ss 00:35:00 -to 00:36:00 ${cam}36.mp4
    ffmpeg -i $videoname -c copy -ss 00:36:00 -to 00:37:00 ${cam}37.mp4
    ffmpeg -i $videoname -c copy -ss 00:37:00 -to 00:38:00 ${cam}38.mp4
    ffmpeg -i $videoname -c copy -ss 00:38:00 -to 00:39:00 ${cam}39.mp4
    ffmpeg -i $videoname -c copy -ss 00:39:00 -to 00:40:00 ${cam}40.mp4
    ffmpeg -i $videoname -c copy -ss 00:40:00 -to 00:41:00 ${cam}41.mp4
    ffmpeg -i $videoname -c copy -ss 00:41:00 -to 00:42:00 ${cam}42.mp4
    ffmpeg -i $videoname -c copy -ss 00:42:00 -to 00:43:00 ${cam}43.mp4
    ffmpeg -i $videoname -c copy -ss 00:43:00 -to 00:44:00 ${cam}44.mp4
    ffmpeg -i $videoname -c copy -ss 00:44:00 -to 00:45:00 ${cam}45.mp4

done



"""