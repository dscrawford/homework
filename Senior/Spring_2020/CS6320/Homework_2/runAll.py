import os

train_dir = 'aclImdb/train'
test_dir = 'aclImdb/test'
executable = 'CS6320_Homework2.py'
for r in ['bow', 'tfidf']:
    for s in ['0','1']:
        if (r == 'tfidf' and s == '1'):
            break
        os.system('python ' + executable + ' ' + train_dir + ' ' + test_dir + ' ' + r + ' nbayes ' + s)
for r in ['bow', 'tfidf']:
    for s in ['0', '1']:
        if (r == 'tfidf' and s == '1'):
            break
        for reg in ['no', 'l1', 'l2']:
            os.system('python ' + executable + ' ' + train_dir + ' ' + test_dir + ' ' + r + ' regression ' + s + ' -r ' + reg)
