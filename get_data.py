import pandas as pd
import csv, os, argparse

def get_args():
	parser = argparse.ArgumentParser(
		description="")
	parser.add_argument("-P", "--preset", dest='preset', type=str,
						help="Choose to use default language set or your own", default="y")

	parser.add_argument("-F", "--Folder", dest='folder', type=str, default="data/raw/",
						help="Directory that contains training data")
	args = parser.parse_args()
	return args

def get_files_from_folder(folder):
	files = os.listdir(folder)

	return folder+files[0], folder+files[1], folder+files[2], folder+files[3], folder+files[4]

def get_languages(csv_file, preset):
	with open(csv_file, 'r') as csv_File: #opens csv containing language codes and their names
		reader = csv.reader(csv_File)
		language_table = {row[0].split(';')[1] : row[0].split(';')[0] for row in reader} #dictionary mapping code to name
	if preset == 'y':
		#Preset language codes to be used in model, chosen at random
		language_codes = ["srd", "krc", "nob", "pnb",
						  "mai", "eng", "be-tarask",
						  "xho", "tet", "tha"]
		language_names = [(key, value) for key, value in language_table.items() if value in language_codes]
		return language_names
	elif preset == 'n':
		'''
		experimental function for allowing user to choose which languages to use
		'''
		languages = []
		while len(languages) != 10:
			language = input("Enter language ").capitalize()
			#print(language)
			if language in languages:
				print("You've already said that one! ")
			elif language in language_table:
				languages.append(language)
				print(' '.join(languages))
			else:
				print('Language not recognised. Please refer to language labels')
				print(' '.join(languages))
				continue
		language_codes = [language_table[i] for i in language_table if i in languages] #collects languages from predetermined set,
		language_names =  [(key, value) for key, value in language_table.items() if value in language_codes] #dictionary mapping code to name
		return language_names
	else:
		print("has to be y or n dummy") #in case of user error

def gen_data(training_file, training_labels, language_codes, training):
	'''
	Function generates the set based on pre defined language codes and creates various
	attributes to the object
	'''
	if training == True:
		data = [i.split('\n')[:-1] for i in open(training_file, 'r')] #opens text file and splits on new line
		labels = [i.split('\n')[:-1] for i in open(training_labels, 'r')] #opens label file and splits on white space
		things = list(zip([i[0] for i in data], [i[0] for i in labels])) #zips sentences with corrosponding language label
		sets = [(i[0],i[1]) for i in things] #this might actually do the same thing as the above not sure

		x = [i[0][:100] for i in sets if i[1] in language_codes] #Matrix of sentences to be used in the model
		y = [i[1] for i in sets if i[1] in language_codes] #labels for each of the sentences
		raw_data = ''.join([i for i in x]) #concatenation of all characters in the training set
		vocab = {char: ord(char) for char in set(raw_data)} #dictionary mapping character to ord(integer)
		int2char = {num : char for char, num in vocab.items()} #dictionary mapping integer to character
		return x, y, vocab, int2char
	else:
		data = [i.split('\n')[:-1] for i in open(training_file, 'r')] #opens text file and splits on new line
		labels = [i.split('\n')[:-1] for i in open(training_labels, 'r')] #opens label file and splits on white space
		things = list(zip([i[0] for i in data], [i[0] for i in labels])) #zips sentences with corrosponding language label
		sets = [(i[0],i[1]) for i in things] #this might actually do the same thing as the above not sure
		x = [i[0][:100] for i in sets if i[1] in language_codes] #Matrix of sentences to be used in the model
		y = [i[1] for i in sets if i[1] in language_codes] #labels for each of the sentences
		return x, y

def output_data(x, y, filename):
	dir = 'data/pre_processed/'

	if os.path.exists(dir) == False:
		os.mkdir(dir)

	output = pd.DataFrame(data={'Language Example' : x, 'Language index' : y})
	pd.DataFrame.to_csv(output, dir+filename)

def get_data_main():
	args = get_args()
	labels, x_test, x_train,y_test,y_train = get_files_from_folder(args.folder)
	language_names = get_languages(labels, args.preset) #arg
	language_codes = [i[1] for i in language_names]
	x_train, y_train, vocab, int2char = gen_data(x_train, y_train, language_codes, training=True)
	x_test, y_test = gen_data(x_test, y_test, language_codes, training=False)
	output_data(x_train,y_train, 'Training_data.csv')
	output_data(x_train,y_train, 'Testing_data.csv')
	return x_train, y_train, vocab, int2char, x_test, y_test

if __name__ == '__main__':
	get_data_main()
