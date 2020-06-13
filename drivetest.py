
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.

drive = GoogleDrive(gauth)

file = drive.CreateFile({'title': "log.txt"})
file.SetContentFile("./log.txt")

file.Upload()
