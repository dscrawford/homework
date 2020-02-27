import os

train_dir = 'aclImdb/train'
test_dir = 'aclImdb/test'
executable = 'CS6320_Homework2.py'
for r in ['bow', 'tfidf']:
    for s in ['0','1']:
        os.system('python ' + executable + ' ' + train_dir + ' ' + test_dir + ' ' + r + ' nbayes ' + s)
for r in ['bow', 'tfidf']:
    for s in ['0','1']:
        for reg in ['no', 'l1', 'l2']:
            os.system('python ' + executable + ' ' + train_dir + ' ' + test_dir + ' ' + r + ' regression ' + s + ' -r ' + reg)
