import time
N = 10000 #number of samples
s3_weight = 0.6

def read_s3_output(file_name):
	file = open('output/' + file_name, 'r')
	all_list = list(file)
	s3_list = all_list[-N:]
	s3_list = [int(x.strip()) for x in s3_list]
	return s3_list

def compute_s3_prob_scores():
	is_pair_probability = [[] for _ in range(N)]
	s3_lists, total_scores = [], []

	score_file = open('output/score.txt', 'r')
	for line in score_file:
		if 'csv_ensemble' in line:
			line = line.strip().replace('\t', ' ')
			[_, file_name, _, _, score] = line.split(' ')
			total_scores.append(float(score))
			s3_lists.append(read_s3_output(file_name))

	average_score = sum(total_scores) / len(total_scores)
	for output_id in range(len(total_scores)):
		s3_score = (total_scores[output_id] - average_score) / s3_weight + 0.5
		print ("id = {}, total_score = {:.4f}, s3_score = {:.4f}"
						.format(output_id + 1, total_scores[output_id], s3_score))

		for sample_id in range(N):
			if s3_lists[output_id][sample_id] == 1:
				is_pair_probability[sample_id].append(s3_score)
			else:
				is_pair_probability[sample_id].append(1 - s3_score)
	print ('average_score = {:.4f}'.format(average_score))

	predicted_file = open('output/s3.txt', 'w')
	s3_prob_scores = [sum(is_pair_probability[sample_id]) / len(is_pair_probability[sample_id])\
							for sample_id in range(N)]
	s3_prob_scores_ = [s for s in s3_prob_scores]
	s3_prob_scores_.sort()
	median_prob_score = s3_prob_scores_[N // 2]
	
	print ('min_prob_score = {:.8f}'.format(min(s3_prob_scores_)))
	print ('median_prob_score = {:.8f}'.format(median_prob_score))
	print ('max_prob_score = {:.8f}'.format(max(s3_prob_scores_)))
	for sample_id in range(N):
		is_pair = int(s3_prob_scores[sample_id] > median_prob_score)
		predicted_file.write('{}\n'.format(is_pair))
	predicted_file.close()

	s3_prob_scores.sort()
	s3_score_file = open('output/s3_score.txt', 'w')
	for sample_id in range(N):
		s3_score_file.write('{:.8f}\n'.format(s3_prob_scores[sample_id]))

	
def ensemble_with_s12():
	file = list(open('output/' + 'submission_mode_3_S1_balance_fold_by_ranking.csv_ensemble.csv.txt', 'r'))
	s12 = file[:(2*N)]

	s3 = list(open('output/' + 's3.txt', 'r'))
	s123 = s12 + s3
	output_file = open('output/s123_{}.txt'.format(int(time.time())), 'w')
	for sample in s123:
		output_file.write(sample)
	output_file.close()
		
		


if __name__ == "__main__":
	compute_s3_prob_scores()
	ensemble_with_s12()