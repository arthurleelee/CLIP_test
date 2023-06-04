import matplotlib.pyplot as plt
def plot(d):
    
    epoch=[]
    loss=[]
    img_acc=[]
    text_acc=[]
    for i in d:
        epoch.append(i[0])
        loss.append(i[1])
        img_acc.append(i[2])
        text_acc.append(i[3])


    plt.plot(epoch,loss)
    plt.xlabel('Epoch') 
    plt.ylabel('loss') 
    plt.title('Training loss')
    plt.savefig('./image/adapter_loss.png')
    #plt.show()
    plt.close()
    
    plt.plot(epoch,img_acc,label='img_acc')
    plt.plot(epoch,text_acc,label='text_acc')
    plt.legend()
    plt.xlabel('Epoch') 
    plt.ylabel('accuracy')
    plt.title('Image and text accuracy')
    plt.savefig('./image/adapter_acc.png')
    #plt.show()
    plt.close()
def parselog(s):
    s1=s.split('|')
    epoch=int(s1[0].split(' ')[-2])
    # print(epoch)
    loss=float(s1[1].split(' ')[-2])
    img_acc=float(s1[2].split(' ')[-2])
    text_acc=float(s1[3].split(' ')[-1][:-1])
    return [epoch,loss,img_acc,text_acc]
def readlog(logfile):
    f= open(logfile, "r")
    data=[]
    for x in f:
        data.append(parselog(x))
    return data[:200]
log1="./adapter_log.txt"
plot(readlog(log1))

