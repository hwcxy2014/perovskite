import numpy
import boto
from boto.s3.connection import S3Connection
from data_io import send_data_to_s3, get_data_from_s3

import sys
sys.path.append("../")

# from constants import s3_bucket_name
from misoLogRegMnistUsps.logreg_mnistusps import LogRegMNISTUSPS

__author__ = 'matthiaspoloczek'

'''
This script takes pickles from my S3, creates a backup in \log_objvals\, transforms all objvals from log(x) to logit(x),
and stores the pickles back into their place
'''

conn = S3Connection()
conn = boto.connect_s3()
bucket = conn.get_bucket('poloczeks3', validate=True)

directory = '/miso/data'
func_name = 'lrMU'
func = LogRegMNISTUSPS(mult=1.0)

def inverse_logit(x):
    return 1.0/(1.0 + numpy.exp(-x))

def logit(x):
    return numpy.log(x) - numpy.log(1.0 - x)

# logit_meanval = numpy.zeros(2)
### Convert pickles for hyper-param optimization with objvals log(testscore) into logit(testscore)
#
# directory_backup = '/miso/data/logarithmic_objvals'
# num_pts = 1000
# for index_IS in func.getList_IS_to_query():
#     key = directory+'/hyper_{1}_IS_{0}_{2}_points'.format(index_IS, func_name, num_pts)
#     data = get_data_from_s3(bucket, key)
#
#     # create backup
#     key_backup = directory_backup + '/hyper_{1}_IS_{0}_{2}_points'.format(index_IS, func_name, num_pts)
#     send_data_to_s3(bucket, key_backup, data)
#
#     # transform values
#     logarithmic_vals = numpy.array(data['vals'])
#     testscore_vals = [(numpy.exp(value + func.getMeanValue(index_IS))) for value in logarithmic_vals]
#     print 'IS '+str(index_IS)+' testscore mean = '+str(numpy.mean(testscore_vals))
#
#     logit_vals = [numpy.log(value) - numpy.log(1 - value) for value in testscore_vals]
#     print 'IS ' + str(index_IS) + ' logit mean = ' + str(numpy.mean(logit_vals))
#
#     logit_meanval[index_IS] = numpy.mean(logit_vals)
#     print logit_meanval[index_IS]
#     data['vals'] = numpy.array([ (value - logit_meanval[index_IS]) for value in logit_vals])
#     print numpy.mean(data['vals'])
#     send_data_to_s3(bucket, key, data)

### Test that data was written correctly
# for index_IS in func.getList_IS_to_query():
#     key = directory+'/hyper_{1}_IS_{0}_{2}_points'.format(index_IS, func_name, num_pts)
#     data = get_data_from_s3(bucket, key)
#
#     print numpy.mean(data['vals'])
#     vals = numpy.array(data['vals'])
#     print numpy.array(data['vals']).shape


### Convert pickles for initial data with objvals log(testscore) into logit(testscore)
#
# directory_backup = '/miso/data/logarithmic_objvals'
# num_replications = 100
# num_pts = 10
# logarithmic_meanval = numpy.array([-1.3149224129624495, -0.82132928759313795])
# logit_meanval = numpy.array([-0.97599, 0.01502])
# for repl_no in xrange(num_replications):
#     print '\nrepl '+str(repl_no)
#     for index_IS in func.getList_IS_to_query():
#         key = directory + '/{2}_IS_{0}_{3}_points_repl_{1}'.format(index_IS, repl_no, func_name, num_pts)
#         data = get_data_from_s3(bucket, key)
#
#         # create backup
#         key_backup = directory_backup + '/{2}_IS_{0}_{3}_points_repl_{1}'.format(index_IS, repl_no, func_name, num_pts)
#         send_data_to_s3(bucket, key_backup, data)
#
#         # transform values
#         logarithmic_vals = numpy.array(data['vals'])
#         testscore_vals = [(numpy.exp(value + logarithmic_meanval[index_IS])) for value in logarithmic_vals]
#         print 'IS '+str(index_IS)+' testscore mean = '+str(numpy.mean(testscore_vals))
#
#         logit_vals = [numpy.log(value) - numpy.log(1 - value) for value in testscore_vals]
#         print 'IS ' + str(index_IS) + ' logit mean = ' + str(numpy.mean(logit_vals))
#
#         print 'logit_meanval[index_IS] = '+str(logit_meanval[index_IS])
#         data['vals'] = numpy.array([ (value - logit_meanval[index_IS]) for value in logit_vals])
#         print 'numpy.mean(data["vals"]) = '+str(numpy.mean(data['vals']))
#         # replace the logarithmic files
#         send_data_to_s3(bucket, key, data)
#
#         # create an additional backup of the files
#         key_backup2 = directory + '/backup/{2}_IS_{0}_{3}_points_repl_{1}'.format(index_IS, repl_no, func_name, num_pts)
#         send_data_to_s3(bucket, key_backup2, data)

### Test that data was written correctly
# repl_no = 99
# num_pts = 10
# for index_IS in func.getList_IS_to_query():
#     key = directory + '/{2}_IS_{0}_{3}_points_repl_{1}'.format(index_IS, repl_no, func_name, num_pts)
#     data = get_data_from_s3(bucket, key)
#
#     print numpy.mean(data['vals'])
#     vals = numpy.array(data['vals'])
#     print numpy.array(data['vals']).shape


### Adapt offset for IS 1, so that it has a real bias on average
# IS1 was corrected by 0.01502, but should have been corrected by -0.97599
# num_pts = 1000
# for index_IS in [1]:
#     key = directory+'/hyper_{1}_IS_{0}_{2}_points'.format(index_IS, func_name, num_pts)
#     data = get_data_from_s3(bucket, key)
#
#     # create backup
#     key_backup = directory_backup + '/hyper_{1}_IS_{0}_{2}_points_wrongOffset'.format(index_IS, func_name, num_pts)
#     send_data_to_s3(bucket, key_backup, data)
#
#     # transform values to testscore for verification:
#     logit_vals_wrongOffset = numpy.array(data['vals'])
#     # print 'IS '+str(index_IS)+' mean logit vals with wrong offset = '+str(numpy.mean(logit_vals_wrongOffset))
#     # testscore_vals = [(inverse_logit(value + 0.01502)) for value in logit_vals_wrongOffset]
#     # print 'IS '+str(index_IS)+' testscore mean = '+str(numpy.mean(testscore_vals))
#     #
#     # logit_vals = [logit(value) for value in testscore_vals]
#     # print 'IS ' + str(index_IS) + ' logit mean = ' + str(numpy.mean(logit_vals))
#     #
#     # data['vals'] = numpy.array([ (value - (-0.97599)) for value in logit_vals])
#     # print 'Conversion via testscores gives mean for logits: '+str(numpy.mean(data['vals']))
#     #
#     # direct_conv_vals = [(value + 0.01502 - (-0.97599)) for value in logit_vals_wrongOffset]
#     # print 'Direct conversion gives mean for logits: ' + str(numpy.mean(direct_conv_vals))
#     #
#     # diff = 0.0
#     # for index in xrange(num_pts):
#     #     diff = numpy.abs(data['vals'][index] - direct_conv_vals[index])
#     # print 'Sum of abs elem-wise diff = ' +str(diff)
#
#     # Direct conversion
#     data['vals'] = [(value + 0.01502 - (-0.97599)) for value in logit_vals_wrongOffset]
#     print 'Direct conversion gives mean for logits: ' + str(numpy.mean(data['vals']))
#     send_data_to_s3(bucket, key, data)


# num_replications = 100
# num_pts = 10
# for repl_no in xrange(num_replications):
#     print '\nrepl '+str(repl_no)
#     for index_IS in [1]:
#
#         key = directory + '/{2}_IS_{0}_{3}_points_repl_{1}'.format(index_IS, repl_no, func_name, num_pts)
#         data = get_data_from_s3(bucket, key)
#         # create backup
#         key_backup = directory_backup + '/{2}_IS_{0}_{3}_points_repl_{1}_wrongOffset'.format(index_IS, repl_no, func_name, num_pts)
#         send_data_to_s3(bucket, key_backup, data)
#
#         logit_vals_wrongOffset = numpy.array(data['vals'])
#         data['vals'] = [(value + 0.01502 - (-0.97599)) for value in logit_vals_wrongOffset]
#         print 'Direct conversion gives mean for logits: ' + str(numpy.mean(data['vals']))
#         send_data_to_s3(bucket, key, data)



### copy back from /backup the files where IS 1 is centered
directory_backup = '/miso/data/backup'
new_func_name = 'lrMU2'
#
# num_pts = 1000
# for index_IS in func.getList_IS_to_query():
#     key = directory_backup+'/hyper_{1}_IS_{0}_{2}_points'.format(index_IS, func_name, num_pts)
#     data = get_data_from_s3(bucket, key)
#
#     # write to lrMU2 file
#     key_new = directory + '/hyper_{1}_IS_{0}_{2}_points'.format(index_IS, new_func_name, num_pts)
#     # print key_new
#     send_data_to_s3(bucket, key_new, data)

# init data
num_replications = 100
num_pts = 10
for repl_no in xrange(num_replications):
    print '\nrepl '+str(repl_no)
    for index_IS in func.getList_IS_to_query():
        key = directory_backup + '/{2}_IS_{0}_{3}_points_repl_{1}'.format(index_IS, repl_no, func_name, num_pts)
        data = get_data_from_s3(bucket, key)

        # write to lrMU2 file
        key_new = directory + '/{2}_IS_{0}_{3}_points_repl_{1}'.format(index_IS, repl_no, new_func_name, num_pts)
        # print key_new
        send_data_to_s3(bucket, key_new, data)

