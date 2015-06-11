import urllib.request
import urllib
import re
import time
import xml
from xml.dom import minidom
import csv

f = open('comedy_comparisons.testtamp')
csv_f = csv.reader(f)

f1= open("funny.txt","w")
f2= open("notfunny.txt","w")
f1d= open("funnyd.txt","w")
f2d= open("notfunnyd.txt","w")

for row in csv_f:
    j=0           
    for i in range(0,2):
        u = "https://gdata.youtube.com/feeds/api/videos/"+row[i].strip()+"?v=2"
        print (u)
        a=0
        while(a<3):
            try:
                url = urllib.request.urlretrieve(u)
                print(url)
                if url[0] == "":
                    continue
                doc = minidom.parse(url[0])
                sentence = doc.getElementsByTagName("title")[0].firstChild.nodeValue
                description = doc.getElementsByTagName("media:description")[0].firstChild.nodeValue
                print(sentence+" "+description)
                if row[2]=="left" and i==0:
                    f1.write(sentence+"\n------------\n")
                    f1d.write(description+"\n------------\n")
                if row[2]=="left" and i==1:
                    f2.write(sentence+"\n------------\n")
                    f2d.write(description+"\n------------\n")
                if row[2]=="right" and i==0:
                    f2.write(sentence+"\n------------\n")
                    f2d.write(description+"\n------------\n")
                if row[2]=="right" and i==1:
                    f1.write(sentence+"\n------------\n")
                    f1d.write(description+"\n------------\n")
                print("SS")
                break
            except:
                print ('retrying')
                a=a+1
f1.flush()
f1.close()
f2.flush()
f2.close()
f1d.flush()
f1d.close()
f2d.flush()
f2d.close()
f.close()
print("Success!!")
            
