import requests
import cv2
import json
import numpy as np
import base64
import time
import tornado.ioloop
import tornado.web
import cnn_net
import os
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

charts = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11,
          'c': 12,
          'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22,
          'n': 23,
          'o': 24, 'p': 25, 'q': 26, 'r': 27, 's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33,
          'y': 34,
          'z': 35
          }


def GetCode():
    try:
        response = requests.get('https://scutspoc.xuetangx.com/api/v1/code/captcha', timeout=10,
                                verify=False)
        content = json.loads(response.text)['data']
        img = content['img']
        captcha_key = content['captcha_key']

        img_data = base64.b64decode(img)
        nparr = np.fromstring(img_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return [img_np, captcha_key]
    except:
        return []

def FenGe(img):
    height, width, _ = np.shape(img)
    number = 0
    state = 0
    charts = np.zeros([4, 2])
    try:
        for i in range(width):
            pixs_2 = img[:, i, 1]
            if (pixs_2 < 64).any():
                if state == 0:
                    if number == 4:
                        raise ValueError
                    charts[number][0] = i
                state = 1
            else:
                if state == 1:
                    charts[number][1] = i
                    number += 1
                state = 0
        if state == 1:
            charts[number][1] = width
        return charts
    except:
        return []


def QuZao(img):
    height, width, _ = np.shape(img)
    for i in range(height):
        for j in range(width):
            pix = img[i, j]
            if (pix <= np.array([64, 64, 64])).all():
                img[i, j] = [255, 255, 255]
    return img


def DrawLine(img, list):
    fenge = []
    height, width, _ = np.shape(img)
    for key, (start, end) in enumerate(list):
        loss = 20 - (end - start)
        if end - start <= 20:
            if (width - np.ceil(end + loss / 2)) < 0:
                _end = width
                _start = width - 20
            else:
                _start = np.ceil(start - loss / 2)
                _end = np.ceil(end + loss / 2)
        else:
            _start = np.ceil(start + loss / 2)
            _end = np.ceil(end - loss / 2)
        fenge.append(img[:, int(_start):int(_end), :])
    return fenge


def Array2Base64(array):
    img_str = cv2.imencode('.jpg', array)[1].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
    b64_code = base64.b64encode(img_str)  # 编码成base64
    return b64_code


def Array2Save(array, name, type=0):
    if type == 0:
        cv2.imwrite('./ground/' + str(name) + '.jpg', array, [int(cv2.IMWRITE_JPEG_CHROMA_QUALITY), 0])
    else:
        cv2.imwrite('./fenge/' + str(name) + '.jpg', array, [int(cv2.IMWRITE_JPEG_CHROMA_QUALITY), 0])


def Rename_pic(id, code):
    ground_pic = cv2.imread('./ground/' + str(id) + '.jpg')
    cv2.imwrite('./ground_mark/' + str(id) + '_' + str(code) + '.jpg', ground_pic,
                [int(cv2.IMWRITE_JPEG_CHROMA_QUALITY), 0])
    for i in range(4):
        fenge = cv2.imread('./fenge/' + str(id) + '_' + str(i) + '.jpg')
        cv2.imwrite('./fenge_mark/' + str(id) + '_' + str(code[i]) + '.jpg', fenge,
                    [int(cv2.IMWRITE_JPEG_CHROMA_QUALITY), 0])
    return


def ListenAndRun():
    app = tornado.web.Application([
        (r'/', MainHandler),
        (r'/upload', Upload),
        (r'/check', Check),
    ])
    app.listen(8000)
    print('准备就绪')
    tornado.ioloop.IOLoop.current().start()


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        callback = self.get_argument('callback', '')
        result = main()
        if result != []:
            self.write(callback + '(' + json.dumps(result) + ')')


class Upload(tornado.web.RequestHandler):
    def get(self):
        callback = self.get_argument('callback', '')
        ID = self.get_argument('ID')
        code = self.get_argument('code')
        if len(code) != 4:
            self.write(callback + '(' + json.dumps(['fail']) + ')')
        else:
            Rename_pic(ID, code)
            self.write(callback + '(' + json.dumps(['success']) + ')')


class Check(tornado.web.RequestHandler):
    def get(self):
        callback = self.get_argument('callback', '')

        pass


def check():
    cnn = cnn_net.cnn_net()
    cnn.saver.restore(cnn.session, cnn.save_path)
    np_train_datas = np.empty(shape=(4, 50, 50, 1), dtype='float32')
    print('准备就绪')

    error = 0
    total = 0
    while True:
        Content = GetCode()
        if Content == []:
            continue
        img = Content[0]

        img = QuZao(img)
        list = FenGe(img)
        fenge = DrawLine(img, list)
        if fenge != []:
            img_list = [cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (50, 50), interpolation=cv2.INTER_CUBIC) for i
                        in
                        fenge]

            for i in range(4):
                np_train_datas[i] = img_list[i][:, :, np.newaxis] / 256  # 归一化g

            predicts = cnn.session.run(cnn.prediction, feed_dict={cnn.x_data: np_train_datas})

            predict = [np.unravel_index(predict.argmax(), predict.shape)[0] for predict in predicts]
            print('测试集预测:', [Int2Chart(i) for i in predict])

            code = ''.join([Int2Chart(i) for i in predict])
            capture_key = Content[1]
            data = {'captcha': code, 'captcha_key': capture_key, 'is_alliance': 0, 'login': '验证码',
                    'password': 'captcha'}
            try:
                response = requests.post('https://scutspoc.xuetangx.com/api/v1/oauth/number/login', data=data,
                                         timeout=5)
                result = response.text
                if '图形验证码错误' in result:
                    name = int(time.time())
                    cv2.imwrite('./error/' + str(name) + '.jpg', img,
                                [int(cv2.IMWRITE_JPEG_CHROMA_QUALITY), 0])
                    error += 1
                total += 1
                print('total: ', total, ' error: ', error, '当前准确率: ', (total - error) / total)
                time.sleep(1)
            except:
                continue


def CheckTrainData():
    cnn = cnn_net.cnn_net()
    cnn.saver.restore(cnn.session, cnn.save_path)
    np_train_datas = np.empty(shape=(4, 50, 50, 1), dtype='float32')
    total = 0
    current = 0
    print('开始测试')
    for path, _, files in os.walk('./ground_mark'):
        for file in files:
            # if (path + '/' + file).find('training') != -1:
            #     continue
            start = time.time()
            file_path = path + '/' + file
            image = cv2.imread(file_path)

            name = file[-8:-4]
            print('测试图片: ', file)
            img = QuZao(image)
            list = FenGe(img)
            fenge = DrawLine(img, list)
            if fenge != []:
                total += 1
                img_list = [cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (50, 50), interpolation=cv2.INTER_CUBIC) for
                            i in
                            fenge]

                for i in range(4):
                    np_train_datas[i] = img_list[i][:, :, np.newaxis] / 256  # 归一化g

                predicts = cnn.session.run(cnn.prediction, feed_dict={cnn.x_data: np_train_datas})

                predict = [np.unravel_index(predict.argmax(), predict.shape)[0] for predict in predicts]

                prediction = ''.join(Int2Chart(i) for i in predict)
                print('预测结果: ', prediction)
                if prediction == name:
                    current += 1
                    print('正确')
                    cv2.imshow('code', img)
                    cv2.waitKey(0)
                else:
                    print('错误')
                print('用时: ', time.time() - start, '秒')
                print('---------------------------------')
    if total != 0:
        print('total：', total, 'current：', current, '整体准确率：', current / total)
    else:
        print('测试集为空')


def Int2Chart(int):
    for i, d in enumerate(charts):
        if i == int:
            return d
    return


def main():
    img = GetCode()[0]
    name = int(time.time())
    img = QuZao(img)
    list = FenGe(img)
    fenge = DrawLine(img, list)
    if fenge != []:
        Array2Save(img, name)
        [Array2Save(i, str(name) + '_' + str(key), 1) for key, i in enumerate(fenge)]
        bs64_quzao = 'data:image/jpeg;base64,' + str(Array2Base64(img), 'utf-8')
        bs64_fenge = ['data:image/jpeg;base64,' + str(Array2Base64(i), 'utf-8') for i in fenge]
        return [name, bs64_quzao, bs64_fenge]
    else:
        return []


if __name__ == '__main__':
    # result = main()
    # print(result)
    check()
    # CheckTrainData()
    # ListenAndRun()
