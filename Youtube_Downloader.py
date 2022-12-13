#!/usr/bin/env python
# coding: utf-8

# In[7]:

#Video Link https://www.youtube.com/watch?v=vEQ8CXFWLZU

from pytube import YouTube
from sys import argv

link = argv[1] # Get Youtube video link as command line argument from user. argv[0] is always the File Name 
yt = YouTube(link)

print("Title: ", yt.title)

print("View: ", yt.views)

yd = yt.streams.get_highest_resolution()

# ADD FOLDER PATH WHERE DOWNLOADED VIDEO WILL BE SAVED
yd.download('D:\Ricky\Data Science\Videos')


# In[ ]:




