import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
# Change these 2 dirs if you store the data somewhere else.
annotations_folder = 'annotations'
videos_folder = 'videos'
class DataLoader():
    def __init__(self, dataset_dic ):
        '''
        :param dataset_dic: A dictionary of dataset folder direction. the keys are the main folders names
        (e.g.: 'bookstore') and the values are a list of sub-folders (e.g."['video0]), [-1] is used when all
        subfolders are included.
        '''
        self.data_dic = dataset_dic
        self.data = []
        self.all_paths = []
        # Name of Objects
        self.objects = np.array([b'"Biker"',
                                 b'"Bus"',
                                 b'"Car"',
                                 b'"Cart"',
                                 b'"Pedestrian"',
                                 b'"Skater"'
                                 ])
        self.set_colors()
        self.load_data()

    def set_colors(self):
        '''
        Assign an specific color to each object
        '''
        np.random.seed(1)
        self.colors = np.random.randint(0,255,(3,6), dtype=np.int32)
        np.random.seed()

    def load_data(self):
        # collect all data paths
        for key in self.data_dic:
            if self.data_dic[key]==[-1]:
                folders = os.listdir(key)
                for folder in folders:
                    self.all_paths.append(os.path.join(key,folder))
            else:
                for folder in self.data_dic[key]:
                    self.all_paths.append(os.path.join(key, folder))
        # Assign a Number to each path and store it in a txt file. (later is used for visualization)
        f = open('dataset_idx.txt','w')
        for idx,path in enumerate(self.all_paths):
            f.write(str(idx)+' '+path+'\n')
        f.close()

        # Load Dataset and encode non-numerical features
        encoder = LabelEncoder()
        encoder.fit(self.objects)
        for idx, path in enumerate(self.all_paths):
            complete_path = os.path.join(annotations_folder,os.path.join(path, 'annotations.txt'))
            print(complete_path,' is loading.')
            raw_data = np.genfromtxt(complete_path, dtype=None )
            temp_data = np.zeros((raw_data.size, 10),dtype=np.int32)
            for column in range(10):
                if column is 9:
                    temp_data[:, column] = encoder.transform(raw_data[:]['f'+ str(column)])
                else:
                    temp_data[:, column] = raw_data[:]['f'+ str(column)]
                #raw_data = raw_data.astype(int)
            self.data.append(temp_data)

    def visualize(self, dataset_idx):
        '''
        :param dataset_idx: the idx of selected data for visualization
        '''
        path = self.all_paths[dataset_idx]
        annotation_data = self.data[dataset_idx]
        video_dir = os.path.join(videos_folder,os.path.join(path,'video.mov'))
        cap = cv2.VideoCapture(video_dir)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('frame width: ', frameWidth)
        print('frame height: ',frameHeight)
        fc = 0
        ret = True

        while ret:
            ret, img = cap.read()
            frame_objects = annotation_data[annotation_data[:,5]==fc,:]
            for i in range(len(frame_objects)):
                color = (int(self.colors[0,frame_objects[i,-1]]),int(self.colors[1,frame_objects[i,-1]]),int(self.colors[2,frame_objects[i,-1]]))
                cv2.rectangle(img, (frame_objects[i,1], frame_objects[i,2]), (frame_objects[i,3], frame_objects[i,4]), color=color, thickness=4)
                cv2.putText(img, str(frame_objects[i,0]), (frame_objects[i,1], frame_objects[i,2]), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            # Press esc to exit
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            fc = fc + 1
            # frame Width and Height is divided by 2 to fit in my screen.
            img = cv2.resize(img, (int(frameWidth/2), int(frameHeight/2)))
            cv2.imshow('BookStore', img)
        cap.release()
        cv2.destroyAllWindows()


if __name__=="__main__":
    data_dic = {'nexus':['video1', 'video2']}
                #'coupa':[-1],
                #'deathCircle':[-1],
                #'gates':[-1],
                #'hyang':[-1],
                #'little':[-1],
                #'nexus':[-1],
                #'quad':[-1]}
    stanford_vis = DataLoader(data_dic)
    stanford_vis.visualize(dataset_idx = 1)