import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Use a service account.
cred = credentials.Certificate('attendance-86c04-firebase-adminsdk-zzn2j-fefdb541b4.json')

app = firebase_admin.initialize_app(cred)

db = firestore.client()