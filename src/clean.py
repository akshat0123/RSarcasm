import json, csv

from tqdm import tqdm


PATHS = '../paths.json'


def main():

    # Load data paths
    paths = json.load(open(PATHS, 'r'))
    paths = { key: '%s%s' % (paths['home'], paths[key]) for key in paths if key != 'home' }

    # Initialize progress bar
    linecount = sum([1 for line in open(paths['raw'], 'r').readlines()])
    progress = tqdm(total=linecount)

    # Process raw comment file
    with open(paths['raw'], 'r') as infile:

        # Read header
        reader = csv.reader(infile, delimiter=',', quotechar='"')
        next(reader)
        progress.update(1)

        # Only keep comment and label for each comment
        with open(paths['clean'], 'w') as outfile: 
            for line in reader:
                label, comment = line[0], line[1]
                outfile.write('%s,"%s"\n' % (label, comment))
                progress.update(1)


if __name__ == '__main__':
    main()
