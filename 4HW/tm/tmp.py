'''
可使用xml.dom.minidom来解析xml文件
'''
import xml.dom.minidom
filename = "./tang300.xml"

class Poem():
    def __init__(self,id:int,  name:str, author:str, content:tuple) -> None:
        self.name = name
        self.author = author
        self.content = content
        self.poem_id = id

    def __str__(self) -> str:
        ans = ''
        ans += self.name + '\n' + self.author + '\n'
        for tmp in self.content:
            ans += tmp + '\n'
        return ans
    
    def get_tuple(self)->tuple:
        ans = []
        ans.append(self.name)
        ans.append(self.author)
        for tmp in self.content:
            ans.append(tmp)
        return tuple(ans)

class Data():
    def __init__(self, file) -> None:
        dom = xml.dom.minidom.parse(file)
        root = dom.documentElement
        content = root.getElementsByTagName('contance')
        line_number = root.getElementsByTagName('line_number')
        Poem_id = root.getElementsByTagName('Poem_id')
        self.data = {}
        self.order = []
        lines = len(content)
        i = 0
        while True:
            #print(line_number[i].childNodes[0].data==str(-100))
            if i == lines:
                break
           

            if line_number[i].childNodes[0].data == str(-100):
                name = content[i].childNodes[0].data[2:]
                author = content[i + 1].childNodes[0].data[2:]
                con = []
                j = i + 2
                while(True):
                    #print(j)
                    try:
                        if line_number[j].childNodes[0].data == str(-100):
                            i = j
                            break
                        else:
                            con.append(content[j].childNodes[0].data)
                            j += 1
                    except:
                        i = j
                        break
                con = tuple(con)
                tmp = Poem(Poem_id[i - 1], name, author, con)
                self.order.append(tmp)
                if list(self.data.keys()).count(author) == 1:
                    self.data[author].append(tmp)
                else:
                    self.data[author] = []
                    self.data[author].append(tmp)
                            
                            
dataset = Data(filename)

#print(names[0].childNodes[0].data)
#print(names[1].childNodes[0].data)
#print(names[2].childNodes[0].data)
# 打印前12行内容
print("前两首诗内容：")
print(dataset.order[0])
print(dataset.order[1])

# get_poems()函数保存20首李白、杜甫、白居易、王維（維是繁体）的诗
def get_poems():
    #TODO
    ans = []
    for i in ['李白','杜甫', '白居易', '王維']:
        for j in range(20):
            ans.append(dataset.data[i][j].get_tuple())
    return ans

print(get_poems()[0])



class BasePublisher(object):
    def __init__(self,name): 
        #pass
        #super.__init__()
        raise NotImplementedError
    def subscribeReader(self, reader):
        
        raise NotImplementedError

    def unsubscribeReader(self, reader):
        raise NotImplementedError

    def notifyReader(self,author, poem):
        raise NotImplementedError
    def __str__(self):
        ans = ''
        ans += '现有的Reader有:\n' 
        for reader in self.reader:
            ans += '    '+reader.name + '\n'
        ans += '现共发布诗歌{}首'.format(self.num) + '\n'
        return ans

class _Publisher(BasePublisher):
    def __init__(self,name):
        self.num = 0
        self.reader = []
        self.num = 0
        self.name = name
        self.reader.clear()
        
        #raise NotImplementedError
    def notifyReader(self,author,poem):
        self.num += 1
        for reader in self.reader:
            reader.receivePoem(self, poem, author)
        #raise NotImplementedError
        


import threading
class BaseReader(object):

    def __init__(self):
        raise NotImplementedError
    def subscribeToPublisher(self, publisher:_Publisher):
        if self.pubisher.count(publisher) == 1:
            print("已经订阅了!")
            return False
        publisher.reader.append(self)
        self.pubisher.append(publisher)
        return True
        raise NotImplementedError
    def unsubscribeToPublisher(self, publisher:_Publisher):
        if self.pubisher.count(publisher) == 0:
            print("已经取消订阅了!")
            return False
        publisher.reader.remove(self)
        self.pubisher.remove(publisher)
        return True
        raise NotImplementedError
    def __str__(self):
        raise NotImplementedError
    def receivePoem(self, publisher, poem , author):
        #print('thread id :{}'.format(threading.current_thread().name)) #如果3.2你使用多线程的话请保留此句，
                                                                       #如果使用多进程的话请类似地输出进程id
        raise NotImplementedError
    def printStatistics(self):
        # 打印消息
        raise NotImplementedError
    

class ReaderType1(BaseReader):

    def __init__(self, _name):
        self.pubisher = []
        self.name = _name
        self.poem ={}
    def receivePoem(self, publisher:_Publisher, poem:tuple, author:str):
        if self.pubisher.count(publisher) == 0:
            return False
        
        if list(self.poem.keys()).count(author) == 1:
            self.poem[author].append(poem)
        else:
            self.poem[author] = []
            self.poem[author].append(poem)
        print('{} received successful.'.format(self.name))
        return True
        #raise NotImplementedError
    def printStatistics(self):
        print("Type1Reader {} has the following Poems:".format(self.name))
        for name in list(self.poem.keys()):
            for t in self.poem[name]:
                print('\n')
                for i in t:
                    print(i)
                    
        #raise NotImplementedError

# 第二种读者
class ReaderType2(BaseReader):

    def __init__(self, _name):
        self.pubisher = []
        self.poem = ()
        self.name = _name
        #raise NotImplementedError

    def receivePoem(self, publisher, poem , author):
        if self.pubisher.count(publisher) == 0:
            return False
        self.poem = poem
        print('{} received successful.'.format(self.name))
        return True
        #raise NotImplementedError

    def printStatistics(self):
        print("Type2Reader {} has the following poem:".format(self.name))
        for i in self.poem:
            print(i)
       #raise NotImplementedError


Publisher = _Publisher("Publisher")
Alice=ReaderType1('Alice')
Bob=ReaderType2('Bob')
Carol=ReaderType2('Carol')


Alice.subscribeToPublisher(Publisher)
Bob.subscribeToPublisher(Publisher)
## 请在这里利用Publisher任意发出几首诗来测试你的代码
lis = get_poems()
Publisher.notifyReader(lis[0][1], lis[0])
Publisher.notifyReader(lis[0][1], lis[1])

Alice.printStatistics()
Bob.printStatistics()
Carol.printStatistics()
print(Publisher)