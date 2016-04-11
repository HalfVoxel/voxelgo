import zipfile

feval = open("evaluate.cpp").read()
fnet = open("net.h").read()

feval = feval.replace('#include "net.h"\n', fnet)

with zipfile.ZipFile("bot.zip", 'w') as z:
	z.writestr("bot.cpp", feval.encode('utf-8'))
