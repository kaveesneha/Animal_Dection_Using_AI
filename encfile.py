import base64
import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet

password_provided = "xyz" # This is input in the form of a string
password = password_provided.encode() # Convert to type bytes
salt = b'salt_' # CHANGE THIS - recommend using a key from os.urandom(16), must be of type bytes
kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=salt,
    iterations=100000,
    backend=default_backend()
)
key = base64.urlsafe_b64encode(kdf.derive(password)) # Can only use kdf once

##key = b'xyz' # Use one of the methods to get a key (it must be the same when decrypting)

###########Encrypt file##############################33
input_file = 'tt.txt'
output_file = 'test.encrypted'
with open(input_file, 'rb') as f:
    data = f.read()

fernet = Fernet(key)
encrypted = fernet.encrypt(data)

with open(output_file, 'wb') as f:
    f.write(encrypted)
# You can delete input_file if you want





###########Decrypt file###############3
##input_file = 'test.encrypted'
##output_file = 'tt2.txt'
##
##with open(input_file, 'rb') as f:
##    data = f.read()
##
##fernet = Fernet(key)
##encrypted = fernet.decrypt(data)
##
##with open(output_file, 'wb') as f:
##    f.write(encrypted)
