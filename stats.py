from collections import Counter

def gen_empty_stats(int2lang, language_names_dict):

    lang_1  = language_names_dict[int2lang[0]]
    lang_2  = language_names_dict[int2lang[1]]
    lang_3  = language_names_dict[int2lang[2]]
    lang_4  = language_names_dict[int2lang[3]]
    lang_5  = language_names_dict[int2lang[4]]
    lang_6  = language_names_dict[int2lang[5]]
    lang_7  = language_names_dict[int2lang[6]]
    lang_8  = language_names_dict[int2lang[7]]
    lang_9  = language_names_dict[int2lang[8]]
    lang_10 = language_names_dict[int2lang[9]]

    language_stats = {lang_1  : {'total_guesses' : 0, 'correct_guesses' : 0, 'incorrect_guesses' : 0, 'languages_guessed' : [], 'num_characters' : []},
                  lang_2  : {'total_guesses' : 0, 'correct_guesses' : 0, 'incorrect_guesses' : 0, 'languages_guessed' : [], 'num_characters' : []},
                  lang_3  : {'total_guesses' : 0, 'correct_guesses' : 0, 'incorrect_guesses' : 0, 'languages_guessed' : [], 'num_characters' : []},
                  lang_4  : {'total_guesses' : 0, 'correct_guesses' : 0, 'incorrect_guesses' : 0, 'languages_guessed' : [], 'num_characters' : []},
                  lang_5  : {'total_guesses' : 0, 'correct_guesses' : 0, 'incorrect_guesses' : 0, 'languages_guessed' : [], 'num_characters' : []},
                  lang_6  : {'total_guesses' : 0, 'correct_guesses' : 0, 'incorrect_guesses' : 0, 'languages_guessed' : [], 'num_characters' : []},
                  lang_7  : {'total_guesses' : 0, 'correct_guesses' : 0, 'incorrect_guesses' : 0, 'languages_guessed' : [], 'num_characters' : []},
                  lang_8  : {'total_guesses' : 0, 'correct_guesses' : 0, 'incorrect_guesses' : 0, 'languages_guessed' : [], 'num_characters' : []},
                  lang_9  : {'total_guesses' : 0, 'correct_guesses' : 0, 'incorrect_guesses' : 0, 'languages_guessed' : [], 'num_characters' : []},
                  lang_10 : {'total_guesses' : 0, 'correct_guesses' : 0, 'incorrect_guesses' : 0, 'languages_guessed' : [], 'num_characters' : []},
                  'total_stats' : {'num_correct' : 0, 'num_incorrect' : 0, 'total_prediction' : 0, 'accuracy' : 0}
                 }
    return language_stats

def update_stats(language_stats,prediction, correct_language, int2lang, characters, language_names_dict):
    language_stats['total_stats']['total_prediction'] += 1
    language_stats[language_names_dict[int2lang[correct_language]]]['total_guesses'] += 1
    language_stats[language_names_dict[int2lang[correct_language]]]['languages_guessed'].append(prediction)
    if prediction == correct_language:
        language_stats[language_names_dict[int2lang[correct_language]]]['correct_guesses'] += 1
        language_stats[language_names_dict[int2lang[correct_language]]]['num_characters'].append(characters)
        language_stats['total_stats']['num_correct'] += 1
    elif prediction != correct_language:
        language_stats[language_names_dict[int2lang[correct_language]]]['incorrect_guesses'] += 1
        language_stats['total_stats']['num_incorrect'] += 1

    language_stats['total_stats']['accuracy'] = round(language_stats['total_stats']['num_correct'] / language_stats['total_stats']['total_prediction'] * 100, 2)
    return language_stats

def further_analysis(language_stats, language_names,int2lang, language_names_dict):
    languages = [i[0] for i in language_names]
    for i in languages:
        lang_guessed = []

        lang_guessed = dict(Counter([int2lang[x] for x in language_stats[i]['languages_guessed']]))
        x = [(value,key) for key, value in lang_guessed.items()]
        sec_max = sorted(x)[-2][1]
        las = sorted(x)[0][1]
        num_char = language_stats[i]['num_characters']
        avg_char = round(sum(num_char) / len(num_char))
        print('Language: {}'.format(i))
        #print('Total guesses: {}'.format(language_stats[i]['total_guesses']))
        print('Total correct: {}'.format(language_stats[i]['correct_guesses']))
<<<<<<< HEAD
<<<<<<< Updated upstream
        print('Total incorrect: {}'.format(language_stats[i]['incorrect_guesses']))
        print('Total accuracy for {}: {}%'.format(i,str(round(language_stats[i]['correct_guesses']/ 500 * 100,2))))

=======
        print('Total accuracy for {}: {}%'.format(i,str(round(language_stats[i]['correct_guesses']/ language_stats[i]['incorrect_guesses'] * 100,2))))
>>>>>>> Stashed changes
=======
        print('Total incorrect: {}'.format(language_stats[i]['incorrect_guesses']))
        print('Total accuracy for {}: {}%'.format(i,str(round(language_stats[i]['correct_guesses']/ 500 * 100,2))))

>>>>>>> b84ebb78d7aa6986346ca5f005c7a721e448e81a
        #print('Languages Guessed: {}'.format(dict(Counter(lang_guessed))))
        #print('Most incorrectly guessed: {}'.format(sec_max))
        #print('Least incorrectly guessed: {}'.format(las))
        print('Average characters until correct guess: {}'.format(avg_char))
        print('\n')
        '''data = {'language'      : i,
                'total_guesses' : language_stats[i]['total_guesses'],
                'total_correct' : language_stats[i]['correct_guesses'],
                'accuracy'      :  str(round(language_stats[i]['correct_guesses']/ language_stats[i]['total guesses'] * 100,2)),
                'languages_guessed' : dict(Counter(lang_guessed))}'''

    def basic_stats(language_stats, language_names,int2lang, language_names_dict):
