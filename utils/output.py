from io import StringIO
import sys
import hashlib

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


def show(output):
    if len(output) == 0:
        print('log is empty')
    else:
        for s in output:
            print(s)

def get_hash(data):
    # Prepare the object hash
    hash_id = hashlib.md5()
    hash_id.update(repr(data).encode('utf-8'))
    return hash_id.hexdigest()
