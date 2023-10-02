import logging
import os,time,random
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import numpy as np
from config import get_common_search_config
import torch
import torch.backends.cudnn as cudnn

from utils import *
from process_data.data_loader import *

from model.Transformer_super_V3 import  Transformer_super_V3
from EMO_public import P_generator,  NDsort,F_distance,F_mating,F_EnvironmentSelect
import matplotlib.pyplot as plt


num_of_Operatopn = 3
num_of_LocalOperatopn = 5

NASdec_baseline = [[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],
                   [2,0,1, 2,0,1,2,0,1,2,0,1],[2,0,1, 2,0,1,2,0,1,2,0,1]]

NASdec_SAINT =    [[1,1,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0],
                   [2,0,1, 2,0,1,2,0,1,2,0,1],[2,0,1, 2,0,1,2,0,1,2,0,1]]

NASdec_SAINT_p =  [[1,1,0,0,0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0,0,0],
                   [2,0,1, 2,0,1,2,0,1,2,0,1],[2,0,1, 2,0,1,2,0,1,2,0,1]]


NASdec_method_1 = [[1,1,0,0,0,1,1,0,0,0,0,0],[0,0,1,1,0,0,0,0,0,0,0,0],
                   [2,0,1, 2,0,1,2,0,1,2,0,1],[2,0,1, 2,0,1,2,0,1,2,0,1]]


NASdec_method_2 = [[1,1,0,0,0,1,1,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0,0,0],
                   [2,0,1, 2,0,1,2,0,1,2,0,1],[2,0,1, 2,0,1,2,0,1,2,0,1]]


Baseline =[NASdec_SAINT,NASdec_SAINT_p,NASdec_method_1,NASdec_method_2]

class individual():
    def __init__(self,dec,length_info=None):
        self.length_info = length_info
        if len(dec)==len(self.length_info):
            self.dec = dec
        else:
            self.deal_IndividualDec_2_NASDec(dec)

        self.get_ndarryDec()
        self.fitness = 0.0

    def get_decF(self):
        A = [12,12,16]
        B = []
        for i,x in enumerate(self.dec):
            if i==0 or i==1 :
                B.append(sum(x))
            elif i==4:
                B.append(sum(sum(x)))
            else:
                continue

        fit = 0.0
        for (x,y) in zip(A,B):

            fit += y/x
        fit /=3
        return 1-fit



    def deal_IndividualDec_2_NASDec(self, individualDec):
        dec = []
        start = 0
        for idx, item in enumerate(self.length_info):
            end = start+np.prod(item)
            dec.append(np.reshape(individualDec[start:end],newshape=item))
            start = end

        # handle decision constraint
        self.dec = []
        for idx, encoding in enumerate(dec):

            if   idx ==0:
                # encoding[:2] = 1
                encoding[0] = 1



            elif idx ==1:
                encoding[2] = 1



            elif idx ==4: # cross-attention
                for item_i in range(encoding.shape[0]):
                    if encoding[item_i].sum()==0:
                        random_idx = np.random.choice(encoding.shape[1], 1)
                        encoding[item_i, random_idx] = 1
                if encoding[:,-1].sum()==0:
                       idxr = np.random.choice(encoding.shape[0],1)
                       encoding[idxr,-1] = 1

            self.dec.append(encoding)

    def get_ndarryDec(self):
        ndarryDec = []
        for item in self.dec:
            ndarryDec.extend(np.array(item).reshape(-1,))
        self.ndarryDec = np.array(ndarryDec)

    def evaluation(self,setting):
        # self.fitness = np.random.rand(2,)
        acc,auc,_ = evaluation(self.dec,setting)
        self.fitness = np.array([auc,0.5])
        # self.fitness = np.array([self.get_decF(),auc])

class EvolutionaryAlgorithm():
    def __init__(self,config):
        self.config = config
        self.load_super_model()
        self.load_validation_dataset()
        self.Popsize = 20
        self.get_Boundary()
        self.Maxi_Gen = 25
        self.gen =0


    def load_super_model(self):
        self.model = Transformer_super_V3(self.config)
        state_dict = torch.load(self.config.pre_train_path)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.config.device)
    def load_validation_dataset(self):
        fold_path = self.config.data_path[0]
        test_data = CTLSTMDataset(config = self.config,mode='test',fold_path=fold_path)
        self.test_dataloader = DataLoader(test_data, batch_size=self.config.batch_size*4, shuffle=False,
                                          drop_last=False, num_workers=self.config.num_workers, collate_fn=test_data.pad_batch_fn)
    def get_Boundary(self):
        # Ednet

        Boundary_Up =  [[1,1,1,1,1,1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1,1,1,1,1,1],
                        [2,2,4,2,2,4,2,2,4,2,2,4],
                        [2,2,4,2,2,4,2,2,4,2,2,4]]

        Boundary_Low = [[0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0]]



        self.dec_length_info=[]
        self.Boundary_Up=[]
        self.Boundary_Low=[]
        for item_up,item_low in zip(Boundary_Up,Boundary_Low):
            self.Boundary_Up.extend(np.array(item_up).reshape(-1,))
            self.Boundary_Low.extend(np.array(item_low).reshape(-1,))

            self.dec_length_info.append(np.array(item_up).shape)

        self.Boundary_Up = np.array(self.Boundary_Up)
        self.Boundary_Low = np.array(self.Boundary_Low)
        self.dec_length = sum([np.prod(x) for x in self.dec_length_info])

        self.SearchSpace = []

        for idx,(i,j) in enumerate(zip(self.Boundary_Up,self.Boundary_Low)):
            data = np.linspace(j,i,i+1).tolist()
            self.SearchSpace.append(data)






    def population_initialization(self):

        self.Population = []
        # self.Population.append(individual([np.array(x) for x in NASdec_baseline],self.dec_length_info))
        # self.Population.append(individual([np.array(x) for x in NASdec_SAINT],self.dec_length_info))
        # self.Population.append(individual([np.array(x) for x in NASdec_SAINT_p],self.dec_length_info))
        # self.Population.append(individual([np.array(x) for x in NASdec_method_1],self.dec_length_info))
        # self.Population.append(individual([np.array(x) for x in NASdec_method_2],self.dec_length_info))

        #self.Population.append(individual(self.Boundary_Low.copy(),self.dec_length_info))

        for i in range(0,self.Popsize):
            prob = (i+1)/(self.Popsize+1)
            dec_i = []
            for j,(up,low) in enumerate(zip(self.Boundary_Up,self.Boundary_Low)):
                # dec_i.extend(np.random.randint(low,up+1,1))

                if j<24:
                    dec_i.extend([int(np.random.rand()<0.1)])
                elif (j+1)%3==0:
                    if np.random.rand()<prob:
                        dec_i.extend([0])
                    else:
                        dec_i.extend(np.random.choice([1,2,3,4],1))
                else:
                    if np.random.rand()<prob:
                        dec_i.extend(np.random.choice([0,2],1))
                    else:
                        dec_i.extend([1])

            dec_i = np.array(dec_i)
            self.Population.append(individual(dec_i,self.dec_length_info))



        self.Pop_fitness = self.Evaluation(self.Population)
        self.set_dir(path='initial')
        self.Save()

    def Evaluation(self, Population):
        Fitness =[]
        for idx,individual_i in enumerate(Population):
            print('Evaluating solution {0}: '.format(idx))
            logging.info('Evaluating solution {0}: '.format(idx))
            # individual_i.evaluation([self.model,self.test_dataloader])
            # Fitness.append(individual_i.fitness)
            Fitness.append([0.78+np.random.rand()/100, 0.5])
        return 1.0-np.array(Fitness)

    def MatingPoolSelection(self):
        self.MatingPool, self.tour_index = F_mating.F_mating(self.Population.copy(), self.FrontValue,
                                                             self.CrowdDistance)

    def deduplication(self,offspring_dec,Parents):
        dedup_offspring_dec = []
        dedup_Parents_dec = []
        for i, j in zip(offspring_dec, Parents):
            if i not in dedup_offspring_dec :
                dedup_offspring_dec.append(i)
                dedup_Parents_dec.append(j)
        return dedup_offspring_dec, dedup_Parents_dec

    def Genetic_operation(self):
        offspring_dec = P_generator.P_generator(self.MatingPool, Boundary=np.vstack([self.Boundary_Up.copy(),self.Boundary_Low.copy()]),
                                                Coding='Binary', MaxOffspring=self.Popsize,SearchSpace=self.SearchSpace)
        # deduplication
        self.offspring = [individual(x,self.dec_length_info) for x in offspring_dec]
        self.off_fitness = self.Evaluation(self.offspring)

    def SearchSpaceReduction(self,Population,Fitness):
        pass
        decs = []
        for indi in Population:
            decs.append(indi.ndarryDec)
        decs = np.array(decs)

        self.spacefitness = []
        self.spacefitnessSTD = []
        self.spacefitnessMinimal = []


        self.spaceLength = []
        for idx,item in enumerate(self.SearchSpace):
            fitness = []
            popDec = decs[:,idx]
            for id in item:
                fitness.extend([sum(Fitness[popDec==id][:,0])/sum(popDec==id)])

            # aviod error
            fitness =np.array(fitness)
            nan_index = np.isnan(fitness)
            fitness[nan_index] = np.mean(fitness[~nan_index])
            fitness = fitness.tolist()
            #----------------------


            self.spacefitness.append(fitness)
            self.spaceLength.append(len(item))


            if len(item)<2:
                self.spacefitnessSTD.append(0)
                self.spacefitnessMinimal.append(0)
            else:
                self.spacefitnessSTD.append(np.std(fitness))
                self.spacefitnessMinimal.append(np.max(fitness))



        # delete max std
        index_set = np.argmax(self.spacefitnessSTD)
        index_num = np.argmax(self.spacefitness[index_set])


        # delete max fitness


        index_set_1 = np.argmax(self.spacefitnessMinimal)
        if index_set_1 == index_set:
            index_set_1 = np.argsort(self.spacefitnessMinimal)[-2]
        index_num_1 = np.argmax(self.spacefitnessMinimal[index_set_1])



        A = self.SearchSpace[index_set].pop(index_num)
        A = self.SearchSpace[index_set_1].pop(index_num_1)











    def EvironmentSelection(self):
        Population = []
        Population.extend(self.Population)
        Population.extend(self.offspring)
        FunctionValue = np.vstack((self.Pop_fitness, self.off_fitness))

        self.SearchSpaceReduction(Population,FunctionValue)

        Population, FunctionValue, FrontValue, CrowdDistance, select_index = F_EnvironmentSelect. \
            F_EnvironmentSelect(Population, FunctionValue, self.Popsize)

        self.Population = Population
        self.Pop_fitness = FunctionValue
        self.FrontValue = FrontValue
        self.CrowdDistance = CrowdDistance
        self.select_index = select_index

    def print_logs(self,since_time=None,initial=False):
        if initial:

            logging.info('********************************************************************Initializing**********************************************')
            print('********************************************************************Initializing**********************************************')
        else:
            used_time = (time.time()-since_time)/60

            logging.info('*******************************************************{0:>2d}/{1:>2d} processing, time spent so far:{2:.2f} min******'
                         '*****************************************'.format(self.gen+1,self.Maxi_Gen,used_time))

            print('*******************************************************{0:>2d}/{1:>2d} processing, time spent so far:{2:.2f} min******'
                  '*****************************************'.format(self.gen+1,self.Maxi_Gen,used_time))

    def set_dir(self,path=None):
        if path is None:
            path = self.gen
        self.whole_path = "{}/Gen_{}/".format(self.config.exp_name, path)

        if not os.path.exists(self.whole_path):
            os.makedirs(self.whole_path)

    def Save(self):
        fitness_file = self.whole_path + 'fitness.txt'
        np.savetxt(fitness_file, self.Pop_fitness, delimiter=' ')
        Pop_file = self.whole_path +'Population.txt'
        SearchSpaceFile = self.whole_path +'Space.txt'
        with open(SearchSpaceFile, "w") as file:
            for j,solution in enumerate(self.SearchSpace):
                file.write(' {}: {} \n '.format(j, solution))


        with open(Pop_file, "w") as file:
            for j,solution in enumerate(self.Population):
                file.write('solution {}: {} \n {} \n {} \n'.format(j, solution.fitness,solution.dec, solution.ndarryDec))

    def Plot(self):
        plt.clf()
        plt.plot(1-self.Pop_fitness[:,0],1-self.Pop_fitness[:,1],'o')
        plt.xlabel('ACC')
        plt.ylabel('AUC')
        plt.title('Generation {0}/{1} \n best ACC: {2:.4f}, best AUC: {3:.4f}'.format(self.gen+1,self.Maxi_Gen,max(1-self.Pop_fitness[:,0]), max(1-self.Pop_fitness[:,1])) )
        # plt.show()
        plt.pause(0.2)
        plt.savefig(self.whole_path+'figure.jpg')

    def main_loop(self):
        plt.ion()
        since_time = time.time()
        self.print_logs(initial=True)
        self.population_initialization()
        self.Plot()

        self.FrontValue = NDsort.NDSort(self.Pop_fitness, self.Popsize)[0]
        self.CrowdDistance = F_distance.F_distance(self.Pop_fitness, self.FrontValue)

        while self.gen<self.Maxi_Gen:
            self.set_dir()
            self.print_logs(since_time=since_time)

            self.MatingPoolSelection()
            self.Genetic_operation()
            self.EvironmentSelection()

            self.Save()
            self.Plot()
            self.gen += 1

        plt.ioff()
        # plt.show()









def continue_training(model,optimizer,dataloader):
    pass


def evaluation(NASdec,setting):

    NASdec.append(np.array([[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1]]))
    model,val_dataloader = setting
    model.eval()
    test_str = 'Validation'

    start_time = time.time()
    output_dict_list = []
    epoch_val_loss = []
    with torch.no_grad():
        total = len(val_dataloader)
        for idx, item in enumerate(val_dataloader):
            output_dict = model.forward(item, NASdec)
            loss = model.loss(output_dict)
            print('\r              [{3} {0:>2d}/{1:>2d}, Loss: {4:.5f}, used_time {2:.2f}min({5:.2f} s)]'.format(idx + 1,total, (time.time() - start_time) / 60,test_str,
                                                                                                                 loss,time.time() - start_time),end='')
            output_dict_list.append(output_dict)
            epoch_val_loss.append(loss.item())

    val_epoch_avg_loss = np.mean(epoch_val_loss)
    metrics = get_metrics(output_dict_list)
    epoch_acc = metrics['acc']
    epoch_auc = metrics['auc']
    cost_time = time.time() - start_time

    print_info = " {3} loss: {0:.5f}, {3} time: {2:.3f}s, metrics: {1}".format(val_epoch_avg_loss, metrics,
                                                                                                cost_time, test_str)
    print(print_info)
    logging.info(print_info)

    return  epoch_acc, epoch_auc,val_epoch_avg_loss



def main():
    # get config
    config = get_common_search_config()
    # device
    config.device, config.device_ids = setup_device(config.n_gpu)

    # fix random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    #----------------Evolutionary Algorithm Search-------------

    EA = EvolutionaryAlgorithm(config)
    EA.main_loop()


    return None






if __name__ == '__main__':
    main()
