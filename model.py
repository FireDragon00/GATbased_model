"""
python = 3.7
pytorch = 1.11.0+cuda11.3
dgl = 0.8.0+cuda11.3
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.glob import SortPooling, AvgPooling, MaxPooling, SumPooling, GlobalAttentionPooling
from dgl.data.utils import load_graphs
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset
from dgl.nn.pytorch.utils import JumpingKnowledge
import pickle

from sklearn.metrics import classification_report
import math

# 选择GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 为当前GPU设置随机种子
SEED = 2
torch.cuda.manual_seed(SEED)

# 图数据的存放目录
data_path = './'
# 参数选择
params = {
"in_feats": 59,
"EPOCHS": 30,
"batch_size": 100,
"hidden_dim": 50,
"lr": 0.001,
"n_heads": 9,
}


# 使用DGL加载完整的数据集
train_graphs, train_graph_labels_dict = load_graphs(data_path + 'train_graphs.bin')
train_graph_labels = train_graph_labels_dict['labels']
test_graphs, test_graph_labels_dict = load_graphs(data_path + 'test_graphs.bin')
test_graph_labels = test_graph_labels_dict['labels']
print('总训练样本数： ', len(train_graphs))
print('测试样本数： ', len(test_graphs))
apps = ['diijam', 'baomoi', 'fptplay', 'iQIYI', 'bigo', 'myradio', 'spotify', 'nhaccuatui', 'soundcloud', 'sachnoiapp',
        'phim247', 'popskid', 'truyenaudiosachnoiviet', 'vieon', 'voizfm', 'tunefm', 'wetv', 'zingmp3', 'truyenaudio',
        'baohay24h',
        'freefire', 'among_us', 'azar', 'comico', 'nimotv', 'mangatoon', 'medoctruyen', 'nhacvang', 'noveltoon',
        'radiofm',
        'vtvgo', 'tivi24h', 'tinder', 'tinmoi24h', 'tivi360', 'tiktok', 'linkedin', 'tiki', 'tinhte', 'lotus',
        'tivi247',
        'tivi_truyentranh_webtoon', 'tuoitre_online', 'vietnamworks', 'wallstreet_journal', 'cnn_news', 'bbc_news',
        'twitter',
        'weeboo', 'twitch', 'vnexpress', 'topcv', 'toc_chien', 'wesing', 'hago', 'google_meet', 'dubsmash', 'facebook',
        'hahalolo',
        'zalo', 'hello_yo', 'dan_tri', 'zoom', 'wikipedia', 'instagram', 'jobway', 'kaka', 'pinterest', 'quora',
        'lazada', 'chess',
        'cake', 'mobile_legend', 'co_tuong_online', 'ted', 'telegram', 'starmarker', 'skype', 'soha', 'tango',
        'thanhnien', 'snapchat',
        'tien_len', 'animal_restaurant', 'bida', 'cho_tot', 'messenger', 'netflix', 'nonolive', 'may', 'podcast_player',
        'pubg',
        'partying', 'kenh14', 'lienquan_mobile', 'likee_lite', 'reddit', 'sendo', 'shopee', 'the_guardian', 'ola_party']
method_app = {
	'Similar': ['diijam', 'myradio', 'spotify', 'nhaccuatui', 'soundcloud', 'sachnoiapp', 'truyenaudiosachnoiviet','voizfm','tunefm', 'radiofm', 'nhacvang', 'wesing', 'kaka', 'podcast_player', 'starmarker', 'zingmp3','truyenaudio'],
	'Different': ['zingmp3', 'fptplay', 'baomoi', 'nimotv', 'messenger', 'tiki', 'facebook', 'lienquan_mobile','quora', 'among_us', 'azar', 'tiktok', 'medoctruyen', 'weeboo', 'tinder', 'hago', 'bida'],
	'10_apps': ['bigo', 'spotify', 'among_us', 'tinder', 'tiktok', 'tiki', 'tuoitre_online', 'hago', 'facebook','netflix'],
	'20_apps': ['bigo', 'spotify', 'freefire', 'among_us', 'azar', 'comico', 'noveltoon', 'tinder', 'tiktok', 'tiki','tuoitre_online', 'wesing', 'hago', 'facebook', 'wikipedia', 'quora', 'snapchat', 'tien_len','messenger', 'netflix'],
	'30_apps': ['baomoi', 'bigo', 'spotify', 'zingmp3', 'freefire', 'among_us', 'azar', 'comico', 'nimotv', 'noveltoon','tinder','tiktok', 'tiki', 'tuoitre_online', 'wesing', 'hago', 'facebook', 'wikipedia', 'instagram', 'pinterest','quora','co_tuong_online', 'ted', 'starmarker', 'snapchat', 'tien_len', 'bida', 'cho_tot', 'messenger','netflix'],
	'40_apps': ['baomoi', 'bigo', 'spotify', 'nhaccuatui', 'zingmp3', 'freefire', 'among_us', 'azar', 'comico', 'nimotv', 'medoctruyen', 'noveltoon', 'tinder', 'tiktok', 'tiki', 'tuoitre_online', 'bbc_news', 'weeboo', 'wesing', 'hago', 'facebook', 'zoom', 'wikipedia', 'instagram', 'pinterest', 'quora', 'co_tuong_online', 'ted', 'starmarker', 'tango', 'snapchat', 'tien_len', 'bida', 'cho_tot', 'messenger', 'netflix', 'nonolive', 'pubg', 'lienquan_mobile', 'reddit'],
	'50_apps': ['baomoi', 'fptplay', 'bigo', 'spotify', 'nhaccuatui', 'soundcloud', 'zingmp3', 'freefire', 'among_us', 'azar', 'comico', 'nimotv', 'medoctruyen', 'noveltoon', 'tinder', 'tiktok', 'tiki', 'lotus', 'tuoitre_online', 'bbc_news', 'twitter', 'weeboo', 'topcv', 'wesing', 'hago', 'google_meet', 'facebook', 'zoom', 'wikipedia', 'instagram', 'pinterest', 'quora', 'chess', 'co_tuong_online', 'ted', 'starmarker', 'tango', 'snapchat', 'tien_len', 'bida', 'cho_tot', 'messenger', 'netflix', 'nonolive', 'pubg', 'lienquan_mobile', 'likee_lite', 'reddit', 'sendo', 'ola_party'
],
	'60_apps': ['baomoi', 'fptplay', 'bigo', 'spotify', 'nhaccuatui', 'soundcloud', 'zingmp3', 'freefire', 'among_us', 'azar', 'comico', 'nimotv', 'mangatoon', 'medoctruyen', 'noveltoon', 'vtvgo', 'tinder', 'tiktok', 'linkedin', 'tiki', 'tinhte', 'lotus', 'tuoitre_online', 'vietnamworks', 'bbc_news', 'twitter', 'weeboo', 'twitch', 'topcv', 'toc_chien', 'wesing', 'hago', 'google_meet', 'dubsmash', 'facebook', 'zoom', 'wikipedia', 'instagram', 'pinterest', 'quora', 'chess', 'co_tuong_online', 'ted', 'starmarker', 'skype', 'tango', 'snapchat', 'tien_len', 'animal_restaurant', 'bida', 'cho_tot', 'messenger', 'netflix', 'nonolive', 'pubg', 'lienquan_mobile', 'likee_lite', 'reddit', 'sendo', 'ola_party'
],
	'70_apps': ['baomoi', 'fptplay', 'bigo', 'spotify', 'nhaccuatui', 'soundcloud', 'wetv', 'zingmp3', 'freefire', 'among_us', 'azar', 'comico', 'nimotv', 'mangatoon', 'medoctruyen', 'noveltoon', 'vtvgo', 'tinder', 'tiktok', 'linkedin', 'tiki', 'tinhte', 'lotus', 'tuoitre_online', 'vietnamworks', 'wallstreet_journal', 'bbc_news', 'twitter', 'weeboo', 'twitch', 'vnexpress', 'topcv', 'toc_chien', 'wesing', 'hago', 'google_meet', 'dubsmash', 'facebook', 'hahalolo', 'hello_yo', 'zoom', 'wikipedia', 'instagram', 'jobway', 'pinterest', 'quora', 'lazada', 'chess', 'cake', 'mobile_legend', 'co_tuong_online', 'ted', 'telegram', 'starmarker', 'skype', 'tango', 'snapchat', 'tien_len', 'animal_restaurant', 'bida', 'cho_tot', 'messenger', 'netflix', 'nonolive', 'pubg', 'lienquan_mobile', 'likee_lite', 'reddit', 'sendo', 'ola_party'
],
	'80_apps': ['baomoi', 'fptplay', 'bigo', 'myradio', 'spotify', 'nhaccuatui', 'soundcloud', 'wetv', 'zingmp3', 'freefire', 'among_us', 'azar', 'comico', 'nimotv', 'mangatoon', 'medoctruyen', 'noveltoon', 'vtvgo', 'tivi24h', 'tinder', 'tivi360', 'tiktok', 'linkedin', 'tiki', 'tinhte', 'lotus', 'tuoitre_online', 'vietnamworks', 'wallstreet_journal', 'bbc_news', 'twitter', 'weeboo', 'twitch', 'vnexpress', 'topcv', 'toc_chien', 'wesing', 'hago', 'google_meet', 'dubsmash', 'facebook', 'hahalolo', 'zalo', 'hello_yo', 'zoom', 'wikipedia', 'instagram', 'jobway', 'kaka', 'pinterest', 'quora', 'lazada', 'chess', 'cake', 'mobile_legend', 'co_tuong_online', 'ted', 'telegram', 'starmarker', 'skype', 'tango', 'thanhnien', 'snapchat', 'tien_len', 'animal_restaurant', 'bida', 'cho_tot', 'messenger', 'netflix', 'nonolive', 'podcast_player', 'pubg', 'partying', 'lienquan_mobile', 'likee_lite', 'reddit', 'sendo', 'shopee', 'the_guardian', 'ola_party'
],
	'90_apps': ['baomoi', 'fptplay', 'bigo', 'myradio', 'spotify', 'nhaccuatui', 'soundcloud', 'phim247', 'popskid', 'voizfm', 'tunefm', 'wetv', 'zingmp3', 'truyenaudio', 'baohay24h', 'freefire', 'among_us', 'azar', 'comico', 'nimotv', 'mangatoon', 'medoctruyen', 'noveltoon', 'tivi247', 'vtvgo', 'tivi24h', 'tinder', 'tinmoi24h', 'tivi360', 'tiktok', 'linkedin', 'tiki', 'tinhte', 'lotus', 'tivi_truyentranh_webtoon', 'tuoitre_online', 'vietnamworks', 'wallstreet_journal', 'cnn_news', 'bbc_news', 'twitter', 'weeboo', 'twitch', 'vnexpress', 'topcv', 'toc_chien', 'wesing', 'hago', 'google_meet', 'dubsmash', 'facebook', 'hahalolo', 'zalo', 'hello_yo', 'zoom', 'wikipedia', 'instagram', 'jobway', 'kaka', 'pinterest', 'quora', 'lazada', 'chess', 'cake', 'mobile_legend', 'co_tuong_online', 'ted', 'telegram', 'starmarker', 'skype', 'tango', 'thanhnien', 'snapchat', 'tien_len', 'animal_restaurant', 'bida', 'cho_tot', 'messenger', 'netflix', 'nonolive', 'podcast_player', 'pubg', 'partying', 'lienquan_mobile', 'likee_lite', 'reddit', 'sendo', 'shopee', 'the_guardian', 'ola_party'
],
	'101_apps': apps}


# 抽样函数，从101个app中抽取10个app、20个app、30个app……的流量图
def sampling(method, graphs, labels):
	SAMPLE = method_app[method]

	apps_dict = {}
	for i, name in enumerate(apps):
		apps_dict[name] = i

	num_label = []
	for app in SAMPLE:
		num_label.append(apps_dict[app])

	new_g = []
	new_l = []
	for label, graph in zip(labels, graphs):
		if label in num_label:
			new_g.append(graph.to(device))  # 转为cuda类型的图，用于GPU计算
			new_l.append(label)

	apps_dict_iname = {}
	for i, name in enumerate(apps):
		apps_dict_iname[i] = name

	x = []
	for l in new_l:
		x.append(l.item())

	SAMPLE_dict = {}
	for i, name in enumerate(SAMPLE):
		SAMPLE_dict[name] = i

	app_name = []
	for l in x:
		app_name.append(apps_dict_iname[l])

	SAMPLE_labels = []
	for name in app_name:
		SAMPLE_labels.append(SAMPLE_dict[name])

	return new_g, torch.tensor(SAMPLE_labels).to(device)


# method_name是抽样个数
method_name = '101_apps'
classes_num = len(method_app[method_name])
BATCH_SIZE = params['batch_size']
EPOCHS = params['EPOCHS']
INIT_LR = params['lr']
# 输入特征维度是59 = 18 * 3 + 1 + 4，18是length statistics，1是protocol field，4是IP field
in_feats = params['in_feats']

# 执行抽样，10个app或20个app等等
new_train_g, new_train_l = sampling(method_name, train_graphs, train_graph_labels)
new_test_g, new_test_l = sampling(method_name, test_graphs, test_graph_labels)
print(method_name, '抽样图数_训练: ', len(new_train_g))
print(method_name, '抽样图数_测试: ', len(new_test_g))
print(method_name, '分类的类数: ', classes_num)


# 符合DGL库的训练集
class Traindataset(DGLDataset):

	def __init__(self, raw_dir=None, force_reload=False, verbose=False):
		super().__init__(name='Train',
		                 raw_dir=raw_dir,
		                 force_reload=force_reload,
		                 verbose=verbose)

	def process(self):
		# 将数据处理为图列表和标签列表
		self.graphs, self.label = new_train_g, new_train_l

	def __getitem__(self, idx):
		return self.graphs[idx], self.label[idx]

	def __len__(self):
		return len(self.graphs)

	@property
	def num_labels(self):
		return classes_num


# 符合DGL库的测试集
class Testdataset(Traindataset):
	def process(self):
		self.graphs, self.label = new_test_g, new_test_l


# 使用DGL加载训练集和测试集
dataset_train = Traindataset()
dataloader_train = GraphDataLoader(dataset_train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
dataset_test = Testdataset()
dataloader_test = GraphDataLoader(dataset_test, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

# 计算一个epoch内的训练批数量和测试批数量
trainSteps = dataset_train.__len__() // BATCH_SIZE
valSteps = dataset_test.__len__() // BATCH_SIZE
print("批大小：", BATCH_SIZE)
print("训练批数量：", trainSteps)
print("测试批数量：", valSteps)


# 模型架构
class Classifier(nn.Module):
	def __init__(self, in_dim, hidden_dim, n_heads, n_classes):
		super(Classifier, self).__init__()
		self.n_heads = n_heads
		self.in_dim = in_dim
		self.hidden_dim = hidden_dim
		self.meanpooling = AvgPooling()

		self.gat1 = GATConv(self.in_dim, self.hidden_dim, self.n_heads)
		self.gat2 = GATConv(self.hidden_dim * self.n_heads, self.hidden_dim, self.n_heads)

		self.tmp = self.hidden_dim * self.n_heads
		self.classify = nn.Linear(self.tmp, n_classes)
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, g, h):
		h = F.relu(self.gat1(g, h))
		h = h.flatten(1)
		h = F.relu(self.gat2(g, h))
		h = h.flatten(1)
		h = self.meanpooling(g, h)
		h = self.classify(h)

		with g.local_scope():
			return self.logsoftmax(h)


# 初始化GAT模型
print("[GNN_Model]初始化图神经网络......")
model = Classifier(in_feats, params['hidden_dim'], params['n_heads'], classes_num)
# 使用GPU
model.to(device)
# 使用Adam优化器
opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)
# 存储每个epoch的训练loss，训练acc，验证loss，验证acc
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}
# 存储训练时长
time_epoch = []
print("[GNN_Model] 训练图神经网络......")
current_acc = 0

# Train the model
for epoch in range(EPOCHS):
	# set the model in training mode
	model.train()

	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0

	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0.00
	valCorrect = 0.00

	startTime_train = time.time()
	for batched_graph, labels in dataloader_train:
		feats = batched_graph.ndata['attr']

		# perform a forward pass and calculate the training loss
		pred = model(batched_graph, feats)
		loss = F.cross_entropy(pred, labels)

		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		opt.zero_grad()
		loss.backward()
		opt.step()

		# add the loss to the total training loss so far
		# and calculate the number of correct predictions
		totalTrainLoss += loss.detach().item()
		trainCorrect += (pred.argmax(1) == labels).type(
			torch.float).sum().item()
	endTime_train = time.time()
	time_epoch.append(endTime_train-startTime_train)

	# switch off autograd for evaluation
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()
		# loop over the validation set
		for batched_graph, labels in dataloader_test:
			feats = batched_graph.ndata['attr']
			# make the predictions and calculate the validation loss
			pred = model(batched_graph, feats)
			loss = F.cross_entropy(pred, labels)

			# calculate the number of correct predictions
			totalValLoss += loss.detach().item()
			valCorrect += (pred.argmax(1) == labels).type(
				torch.float).sum().item()

	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps
	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / (trainSteps * BATCH_SIZE)
	valCorrect = valCorrect / (valSteps * BATCH_SIZE)

	# update our training history
	H["train_loss"].append(avgTrainLoss)
	H["train_acc"].append(trainCorrect)
	H["val_loss"].append(avgValLoss)
	H["val_acc"].append(valCorrect)

	# print the model training and validation information
	print("[GAT_Model] EPOCH: {}/{}-{}s".format(epoch + 1, EPOCHS, int(endTime_train - startTime_train)))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}, Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
		avgTrainLoss, trainCorrect, avgValLoss, valCorrect))

	# 保存最优结果
	if current_acc < valCorrect:
		current_acc = valCorrect
		torch.save(model.state_dict(), './model_params.pt')

# Eval the model
model.load_state_dict(torch.load('./model_params.pt'))
model.eval()
model.to(device)
with torch.no_grad():
	pred = []
	true = []
	for batched_graph, labels in dataloader_test:
		feats = batched_graph.ndata['attr']
		outputs = model(batched_graph, feats)
		_, predicted = torch.max(outputs.data, 1)
		pred.extend(predicted.cuda().data.cpu().numpy())
		true.extend(labels.cuda().data.cpu().numpy())

print(classification_report(true, pred, target_names=method_app[method_name], digits=4))


# 保存运行时间，每轮分类精度
file = open('./train_test.pickle', 'wb')
pickle.dump(H, file)
file.close()
file = open('./train_time.pickle', 'wb')
pickle.dump(time_epoch, file)
file.close()
