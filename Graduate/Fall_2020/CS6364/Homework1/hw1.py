# Made by Daniel Crawford
# Student Net ID: dsc160130
# Course: CS6364 - Artificial Intelligence
import datetime
import json
import os

from pattern.en import positive, sentiment
from numpy import unique


def task1():
    print('hello world')


def task2():
    items = [1, 2, 3, 4, 5]
    print(items)


def task3():
    items = open('task3.data').read().split(' ')
    items1, items2 = items[:len(items) // 2], items[len(items) // 2:]
    print('items1: ', items1, '\n',
          'items2: ', items2, sep='')


def task4(data):
    data = {'school': 'UAlbany', 'address': '1400 Washington Ave, Albany, NY 12222', 'phone': '(518) 442-3300'}
    for key, item in data.items():
        print(key + ': ' + item, sep='')


def task5(data, file_name):
    f = open(file_name, 'w')
    json.dump(data, f)

    f = open(file_name, 'r')
    d = json.load(f)
    for key, item in d.items():
        print(key, item)

    f.close()


def task6(data):
    # list
    items = [1, 2, 3, 4, 5]
    f = open('items.data', 'w')
    json.dump(items, f)
    f = open('items.data', 'r')
    print(json.load(f))

    # dict
    f = open('task6.data', 'w')
    json.dump(data, f)
    f = open('task6.data', 'r')
    print(json.load(f))

    f.close()


def task7(file_name):
    tweets = [json.loads(line) for line in open(file_name).readlines()]
    for tweet in tweets:
        print(tweet['id'])


def task8(file_name):
    tweets = [json.loads(line) for line in open(file_name).readlines()]
    sorted_tweets = sorted(tweets, key=lambda item: datetime.datetime.strptime(item['created_at'],
                                                                               '%a  %b  %d  %H:%M:%S  +0000  %Y'))
    f = open('task8.data', 'w')
    f.write('[' + '\n')
    for tweet in sorted_tweets[-10:]:
        f.write(json.dumps(tweet) + ',\n')
    f.write(']')
    print('Wrote out top 10 most recent tweets to task8.data')
    f.close()


def task9(file_name):
    tweets = [json.loads(line) for line in open(file_name).readlines()]
    for tweet in tweets:
        date = datetime.datetime.strptime(tweet['created_at'], '%a  %b  %d  %H:%M:%S  +0000  %Y')
        tweet['shortened_date'] = str(date.month) + '-' + str(date.day) + '-' + str(date.year) + '-' + str(date.hour)

    os.makedirs('task9-output', exist_ok=True)

    all_dates = unique([tweet['shortened_date'] for tweet in tweets])
    grouped_dates = {date: [] for date in all_dates}

    for tweet in tweets:
        grouped_dates[tweet['shortened_date']].append(tweet)

    for mdyh, subtweets in grouped_dates.items():
        f = open('./task9-output/' + mdyh + '.txt', 'w')
        json.dump(subtweets, f)
        f.close()

    print('Wrote out grouped tweets by Mon-Day-Year-Hour in task9-output')


def task10(file_name):
    tweets = [json.loads(line) for line in open(file_name).readlines()]
    sentiments = [positive(tweet['text'], threshold=0.1) for tweet in tweets]
    negative_tweets, positive_tweets = [tweets[i] for i, pos in enumerate(sentiments) if not pos], \
                                       [tweets[i] for i, pos in enumerate(sentiments) if pos]

    print('Sentiment Scores')
    for tweet in tweets:
        sent = sentiment(tweet['text'])
        print('Polarity: ', sent[0], ', Subjectivity: ', sent[1])

    f = open('positive-sentiment-tweets.txt', 'w')
    json.dump(positive_tweets, f)

    f = open('negative-sentiment-tweets.txt', 'w')
    json.dump(negative_tweets, f)

    f.close()

    print('Wrote out positive and negative sentiment tweets in positive-sentiment-tweets.txt and '
          'negative-sentiment-tweets.txt')


data = {'school': 'UAlbany', 'address': '1400 Washington Ave, Albany, NY 12222', 'phone': '(518) 442-3300'}
if __name__ == '__main__':
    tasks = [
        (task1, {}),
        (task2, {}),
        (task3, {}),
        (task4, {'data': data}),
        (task5, {'data': data, 'file_name': 'data.json'}),
        (task6, {'data': data}),
        (task7, {'file_name': 'CrimeReport.txt'}),
        (task8, {'file_name': 'CrimeReport.txt'}),
        (task9, {'file_name': 'CrimeReport.txt'}),
        (task10, {'file_name': 'CrimeReport.txt'}),
    ]

    i = 1
    for task, args in tasks:
        print('task', i, ': ', )
        task(**args)
        print()
        i += 1
