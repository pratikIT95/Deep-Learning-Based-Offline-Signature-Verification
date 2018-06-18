import os
filestodel=os.listdir("C:\\Users\\HP\\Desktop\\Project Stuff\\Code\\Test\\Training Set")
i=0
destdir="C:\\Users\\HP\\Desktop\\Project Stuff\\Signature Dataset\\Testdata_SigComp2011\\SigComp11-Offlinetestset\\Dutch\\Lol"
for i in range(0,len(filestodel)):
    print("Deleting file:",destdir+"\\"+filestodel[i])
    os.remove(destdir+"\\"+filestodel[i])
