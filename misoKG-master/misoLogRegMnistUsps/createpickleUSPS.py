import numpy
import csv
import cPickle as pickle

# read the USPS data. This are 9298 rows
train_vector_b = numpy.zeros(5298, dtype=numpy.int64)
train_matrix_x = numpy.zeros((5298, 256), dtype=numpy.float32)
valid_vector_b = numpy.zeros(2000, dtype=numpy.int64)
valid_matrix_x = numpy.zeros((2000, 256), dtype=numpy.float32)
test_vector_b = numpy.zeros(2000, dtype=numpy.int64)
test_matrix_x = numpy.zeros((2000, 256), dtype=numpy.float32)

# randomly select rows for validation set, in response to criticism for USPS dataset
# array_validation_rows = numpy.random.choice(9298, 2007, replace=False)
allrows = [x for x in xrange(9298)]
numpy.random.shuffle(allrows)
array_validation_rows = allrows[:2000]
array_test_rows = allrows[2000:4000]
#
# print len(allrows)
# print len(array_validation_rows)
# print len(array_test_rows)
# print array_test_rows[0]
# print array_validation_rows[0]
# exit(0)

with open('/home/ubuntu/data/usps.csv', 'rb') as f:
    reader = csv.reader(f)
    # print reader.line_num

    no_next_train = 0
    no_next_valid = 0
    no_next_test = 0
    rowno = 0
    for row in reader:
        # print row[1]
        # print len(row)
        if rowno in array_validation_rows:
            valid_vector_b[no_next_valid] = int(row[0]) -1
            # The csv file has incremented the observed class, i.e. digit i becomes i+1
            for index in xrange(1,256):
                valid_matrix_x[no_next_valid, index-1] = row[index]
            no_next_valid += 1
        elif rowno in array_test_rows:
            test_vector_b[no_next_test] = int(row[0]) -1
            for index in xrange(1,256):
                test_matrix_x[no_next_test, index-1] = row[index]
            no_next_test += 1
        else:
            train_vector_b[no_next_train] = int(row[0]) -1
            for index in xrange(1,256):
                train_matrix_x[no_next_train, index-1] = row[index]
            no_next_train += 1
        # vector_b[rowno] = row[0]
        # for index in xrange(1,256):
        #     matrix_x[rowno, index-1] = row[index]
        rowno += 1

with open('/home/ubuntu/data/usps.pickle', "wb+") as output_file:
    pickle.dump( ((train_matrix_x,train_vector_b), (valid_matrix_x,valid_vector_b),
                  (test_matrix_x,test_vector_b)
                  ), output_file )

