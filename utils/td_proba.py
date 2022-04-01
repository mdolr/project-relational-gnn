word = 'aaa'


def index(w):
    if len(w) == 0:
        return 0
    else:
        start = (len(w) // 2) - 1
        end = len(w) // 2
        val = end - start

        j = 0

        while (j < len(w)):
            if w[j] == 'a':
