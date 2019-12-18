import pandas as pd

pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import lag_plot
import numpy as np
import calendar
from tqdm import tqdm
import collections



def read_data(rows):
    print('Reading train.csv file....')
    train = pd.read_csv('C:/PYTHON/Kaggle/2019 Data Science Bowl/data/train.csv', nrows=rows)
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))
    train_drop = train
    keep_id = train_drop[train_drop.type == "Assessment"]['installation_id'].drop_duplicates()
    train = pd.merge(train, keep_id, on="installation_id", how="inner")



    print('Reading test.csv file....')
    test = pd.read_csv('C:/PYTHON/Kaggle/2019 Data Science Bowl/data/test.csv', nrows=rows)
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('C:/PYTHON/Kaggle/2019 Data Science Bowl/data/train_labels.csv', nrows=rows)
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('C:/PYTHON/Kaggle/2019 Data Science Bowl/data/specs.csv', nrows=rows)
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('C:/PYTHON/Kaggle/2019 Data Science Bowl/data/sample_submission.csv', nrows=rows)
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0],
                                                                          sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission


def encode_title(train, test, train_labels):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    list_of_type_code = list(set(train['type'].unique()).union(set(test['type'].unique())))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))

    type_activities = dict(zip(list_of_type_code, np.arange(len(list_of_type_code))))

    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(
        set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    #train['type']=test['type'].map(type_activities)

    win_code = dict(zip(activities_map.values(), (4100 * np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code






def get_train_and_test(train, test):
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
        compiled_train += get_data(user_sample)
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
        test_data = get_data(user_sample, test_set = True)
        compiled_test.append(test_data)
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    categoricals = ['session_title']
    return reduce_train, reduce_test, categoricals

def reduce_zero_time(train,  flag_event_train):
    num0 = 0
    num1 = 0
    num_z = 0

    day = pd.DataFrame()
    day[['date', 'time']] = train['timestamp'].str.split('T', expand=True)
    day[['hour', 'minut', 'sec']] = day['time'].str.split(':', expand=True)
    day[['hour']] = day[['hour']].astype(int)

    ser_hour = pd.Series(day['hour'], index=day.index)
    ser_minut = pd.Series(day['minut'], index=day.index)
    Session_time = pd.DataFrame({"game_session": [0], "long_time": [0], "ID_num0": [0], "tipe": [0]})
    Session_time.loc[[0], ['game_session']] = train.iloc[num0, 1]
    Session_time.loc[[0], ['ID_num0']] = num0
    Session_time.loc[[num1], ['tipe']] = train.iloc[0, 9]
    while (num0 < (train.shape[0] - 1)):
        delta_time0 = pd.to_numeric(ser_hour[num0]) * 60 + pd.to_numeric(ser_minut[num0])

        while (Session_time.iloc[num1, 0] == train.iloc[num0, 1] and num0 < (train.shape[0] - 1)):
            num0 = num0 + 1
            # 1 Записать время в long_time
            # 2 Прописать дельту
            # 3 расчитать время сезона

        delta_time = pd.to_numeric(ser_hour[num0 - 1]) * 60 + pd.to_numeric(ser_minut[num0 - 1]) - delta_time0
        if (delta_time < 0):
            delta_time = pd.to_numeric(ser_hour[num0 - 1]) * 60 + pd.to_numeric(
                ser_minut[num0 - 1]) + 1440 - delta_time0
        if (delta_time < 1):
            num_z = num_z + 1
            # train = pd.merge(train, train.loc[[num0], ['game_session']], on="game_session", how="inner")

        print(num0)
        Session_time.loc[[num1], ['long_time']] = delta_time
        Session_time = Session_time.append({'game_session': train.iloc[num0, 1]}, ignore_index=True)
        num1 = num1 + 1
        Session_time.loc[[num1], ['ID_num0']] = num0
        Session_time.loc[[num1], ['tipe']] = train.iloc[num0, 9]

    if(flag_event_train==0):

        keep_ltime = Session_time[Session_time.long_time > 0.1]['game_session'].drop_duplicates()
        keep_ltime_1 = Session_time[Session_time.long_time < 100]['game_session'].drop_duplicates()
        keep_ltime=pd.merge(keep_ltime, keep_ltime_1,  how="inner")
        train = pd.merge(train, keep_ltime, on="game_session", how="inner")
        return train
    else:
       return Session_time


def preprocess(reduce_train, reduce_test):
    for df in [reduce_train, reduce_test]:
        #df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')
        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')
        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')

        df['sum_event_code_count'] = df[
            [2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020,
             4021,
             4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080,
             2035,
             2040, 4090, 4220, 4095]].sum(axis=1)

        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform(
            'mean')

    features = reduce_train.loc[
        (reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns  # delete useless columns
    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in
                                                                                          assess_titles]

    return reduce_train, reduce_test, features


def get_data(user_sample, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_activity = 0
    user_activities_count = {'Clip': 0, 'Activity': 0, 'Assessment': 0, 'Game': 0}

    # news features: time spent in each activity
    time_spent_each_act = {actv: 0 for actv in list_of_user_activities}
    event_code_count = {eve: 0 for eve in list_of_event_code}
    last_session_time_sec = 0

    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []

    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session

        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]  # from Andrew

        # get current session time in seconds
        if session_type != 'Assessment':
            time_spent = int(session['game_time'].iloc[-1] / 1000)
            time_spent_each_act[activities_labels[session_title]] += time_spent

        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session) > 1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens:
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(time_spent_each_act.copy())
            features.update(event_code_count.copy())
            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]  # from Andrew
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts
            accumulated_uncorrect_attempts += false_attempts
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy / counter if counter > 0 else 0
            accuracy = true_attempts / (true_attempts + false_attempts) if (true_attempts + false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group / counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions

            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts + false_attempts > 0:
                all_assessments.append(features)

            counter += 1

        # this piece counts how many actions was made in each event_code so far
        n_of_event_codes = collections.Counter(session['event_code'])

        for key in n_of_event_codes.keys():
            event_code_count[key] += n_of_event_codes[key]

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type
    # if test_set=True, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in train_set, all assessments are kept
    return all_assessments




day=pd.DataFrame()

train, test, train_labels, specs, sample_submission=read_data(rows=100000)
#train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code=encode_title(train, test, train_labels)

day[['date','time']] = train['timestamp'].str.split('T',expand=True)
day[['hour','minut', 'sec']]=day['time'].str.split(':',expand=True)
day[['hour']]=day[['hour']].astype(int)


ser_data=pd.Series(day['date'], index=day.index)
ser_hour=pd.Series(day['hour'], index=day.index)
ser_minut=pd.Series(day['minut'], index=day.index)

train=reduce_zero_time(train, flag_event_train=0)
Session_time=reduce_zero_time(train, flag_event_train=1)

train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code=encode_title(train, test, train_labels)
print('Post_clear_Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))
sample_id = train[train.installation_id == "0006a69f"]
sample_id_data = get_data(sample_id) #returns a list
sample_df = pd.DataFrame(sample_id_data)
#train, test= preprocess(train, test)

new_test = []
for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=test.installation_id.nunique(),
                                desc='Installation_id', position=0):
    a = get_data(user_sample, test_set=True)
    new_test.append(a)

reduce_test = pd.DataFrame(new_test)

print("Finish")
