import sys,json,urllib.request
url,label=sys.argv[1],sys.argv[2]
for i,l in enumerate(open('tricky.txt').read().splitlines()):
    r=urllib.request.Request(url+"/tts",data=json.dumps({"text":l,"voice":"af_heart"}).encode(),headers={"Content-Type":"application/json"})
    open(f"asr/{label}_{i}.wav","wb").write(urllib.request.urlopen(r,timeout=300).read())
