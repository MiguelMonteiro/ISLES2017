import os
from PostProcessing.Utils import CRF, adjust_training_data

os.chdir('../')

sdims_values = [50]
schan_values = [None]
output = []
for sdims in sdims_values:
    for schan in schan_values:
        print('sdims: {0}, schan: {1}'.format(str(sdims), str(schan)))

        crf = CRF(sdims, None)

        def transform(prediction, probability, image):
            return crf.adjust_prediction(probability, image, iter=5)
        metrics = adjust_training_data(transform)
        output.append([(sdims, schan), metrics])


# with open('output.pickle', 'w') as f:
#     pickle.dump(output, f)
#
# with open('output.pickle', 'r') as f:
#     output = pickle.load(f)
#
#
# def sort_function(item):
#     pre = np.mean(item[1]['dc']['pre_crf'])
#     post = np.mean(item[1]['dc']['post_crf'])
#     return (post-pre)/pre*100
#
#
# sorted_ouput = sorted(output, key=sort_function)
#
# for i in sorted_ouput:
#     print i[0]
#     print('dc')
#     report_metric(i[1]['dc']['pre_crf'], i[1]['dc']['post_crf'])
#     print('hd')
#     report_metric(i[1]['hd']['pre_crf'], i[1]['hd']['post_crf'])
#     print('assd')
#     report_metric(i[1]['assd']['pre_crf'], i[1]['assd']['post_crf'])
