import requests
import json
import time
import cv2
import math
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import threading
import cnn_net
import numpy as np
import base64
from code_pre import QuZao, DrawLine, FenGe, Int2Chart

# 禁用安全请求警告
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class test:
    def __init__(self):
        self.cookie = requests.cookies.RequestsCookieJar()
        self.course = ''
        self.userInfo = ''
        self.classroom = ''
        self.videos = []
        self.record = ''
        self.unfinish = []
        self.username = ''
        self.password = ''

    def getCourseId(self):
        if self.course == '':
            course = '8761'
            self.course = course
        return self.course

    def getUserInfo(self):
        if self.userInfo == '':
            url = 'https://scutspoc.xuetangx.com/header_ajax'
            response = requests.get(url, cookies=self.cookie, verify=False)
            response.encoding = response.apparent_encoding
            Json = json.loads(response.text)
            self.userInfo = Json['userInfo']
            return self.userInfo
        else:
            return self.userInfo

    def getClassroom(self):
        if self.classroom == '':
            url = 'https://scutspoc.xuetangx.com/lms/api/v1/course/' + self.getCourseId() + '/about'
            response = requests.get(url, cookies=self.cookie, verify=False)
            response.encoding = response.apparent_encoding
            Json = json.loads(response.text)
            self.classroom = str(Json['data']['classes'][0]['id'])
            return self.classroom
        else:
            return self.classroom

    def getVideos(self):
        if self.videos == []:
            url = 'https://scutspoc.xuetangx.com/score/' + self.getCourseId() + '/manual_paging/'
            response = requests.post(url, data={'cla': 'v', 'class_id': self.getClassroom(), 'cp': '', 'page': 1,
                                                'status': -1}, cookies=self.cookie,
                                     verify=False)
            response.encoding = response.apparent_encoding
            Json = json.loads(response.text)
            self.videos = Json['data']['items']

            total = Json['data']['count']
            number = 2
            while total is not None and total > len(self.videos):
                url = 'https://scutspoc.xuetangx.com/score/' + self.getCourseId() + '/manual_paging/'
                response = requests.post(url,
                                         data={'cla': 'v', 'class_id': self.getClassroom(), 'cp': '', 'page': number,
                                               'status': -1}, cookies=self.cookie,
                                         verify=False)
                response.encoding = response.apparent_encoding
                Json = json.loads(response.text)
                self.videos = self.videos + [i for i in Json['data']['items'] if i['name'] == 'Video']
                number += 1
            print('\n视频列表:\n')

            for video in self.videos:
                print(video['chapter'], ' | ', video['section'], ' | ', video['percent'], '%')
            return self.videos
        else:
            return self.videos

    def getLastRecord(self):
        url = 'https://scutspoc.xuetangx.com/video_point/get_video_watched_record?cid=' + self.getCourseId() + '&vtype=rate&video_type=video'
        response = requests.get(url, cookies=self.cookie, verify=False)
        response.encoding = response.apparent_encoding
        Json = json.loads(response.text)
        self.record = Json
        return self.record

    def GetVideoUrl(self, video_id):
        url = 'https://scutspoc.xuetangx.com/server/api/class_videos/?video_id=' + str(video_id) + '&class_id=' + str(
            self.getClassroom())
        response = requests.get(url, cookies=self.cookie, verify=False)
        response.encoding = response.apparent_encoding
        Json = json.loads(response.text)
        return Json['video_playurl']['quality10'][0]

    def loadRecordToVideos(self):
        videos = self.getVideos()
        print('\n本次播放的视频：\n')
        last = self.getLastRecord()
        for key, video in enumerate(videos):
            if video['item_id'] in last:
                if video['percent'] != 100:
                    video['last'] = last[video['item_id']]['last_point']
                    video['length'] = last[video['item_id']]['video_length']
                    self.unfinish.append(video)
                    print(video['chapter'], ' | ', video['section'], ' | ', video['percent'], '%')
            else:
                self.unfinish.append(video)
                print(video['chapter'], ' | ', video['section'], ' | ', video['percent'], '%')
        if self.unfinish == []:
            print('你已完成所有视频!')
            print('自动退出脚本')
        return self.unfinish

    def getVideoGroup(self):
        return 'sjy'

    def proccess(self, video_id, last, name):
        status = self.CheckCookie()
        if status is False:
            return False
        self.getStatus()
        sq = 1
        self.LoadStart(video_id, sq)
        self.StudyRecord(video_id)
        sq += 1
        self.Stalled(video_id, sq)
        total = self.GetLong(self.GetVideoUrl(video_id))
        sq += 1
        self.Seeking(video_id, last, total, sq)
        sq += 1
        self.Loadeddata(video_id, last, total, sq)
        time.sleep(0.5)
        sq += 1
        self.play(video_id, last, total, sq)
        time.sleep(0.5)
        sq += 1
        self.waiting(video_id, last, total, sq)
        time.sleep(0.5)
        sq += 1
        self.playing(video_id, last, total, sq)
        for i in range(0, int(total * 10), 50):
            print('%+50s' % (name,) + ' | cp: ' + str(i / 10) + ' | total: ' + str(total))
            sq += 1
            self.heartbeat(video_id, i / 10, last, total, sq)
            sq += 1
            self.Pause(video_id, i / 10, last, total, sq)
            sq += 1
            self.play(video_id, last, total, sq, i / 10)
            sq += 1
            self.playing(video_id, last, total, sq, i / 10)
            if i % 100 == 0:
                status = self.CheckCookie()
                if status is False:
                    return False
            time.sleep(5)
        time.sleep(3)
        sq += 1
        self.Pause(video_id, total, last, total, sq)
        sq += 1
        self.videoend(video_id, last, total, sq)
        return True

    def GetVersion(self):
        url = 'https://scutspoc.xuetangx.com/lms/api/v1/course/' + self.getCourseId() + '/sample_about?class_id=' + self.getClassroom()
        resposne = requests.get(url, cookies=self.cookie, verify=False)
        resposne.encoding = resposne.apparent_encoding
        Json = json.loads(resposne.text)
        return Json['data']['version_id']

    def StudyRecord(self, video_id):
        url = 'https://scutspoc.xuetangx.com/lms/api/v1/study_record/'
        resposne = requests.post(url, {'course_id': self.getCourseId(),
                                       'item_id': video_id,
                                       'type': "0",
                                       'version_id': self.GetVersion()}, cookies=self.cookie, verify=False)
        return True

    def getStatus(self):
        url = 'https://scutspoc.xuetangx.com/live_cast/list/?status=ing&show=true&cast_type=0'
        response = requests.get(url, cookies=self.cookie, verify=False)
        url2 = 'https://scutspoc.xuetangx.com/lms/api/v1/course_manage/student_course_has_ing_live_cast'
        response2 = requests.get(url2, cookies=self.cookie, verify=False)
        url3 = 'https://scutspoc.xuetangx.com/lms/api/v1/course/' + self.getCourseId() + '/courseware/'
        response3 = requests.post(url3, {'class_id': self.getClassroom()}, cookies=self.cookie, verify=False)
        return True

    def rateChange(self, video_id, sq):
        url = 'https://scutspoc.xuetangx.com/heartbeat?i=5&et=heartbeat'
        response = requests.get(url, self.Params('ratechange', video_id, 0, 0, 0, sq), cookies=self.cookie,
                                verify=False)
        print('发送ratechange包')
        return True

    def LoadStart(self, video_id, sq):
        url = 'https://scutspoc.xuetangx.com/heartbeat?i=5&et=heartbeat'
        response = requests.get(url, self.Params('loadstart', video_id, 0, 0, 0, sq), cookies=self.cookie, verify=False)
        print('发送loadstart包')
        return True

    def Stalled(self, video_id, sq):
        url = 'https://scutspoc.xuetangx.com/heartbeat?i=5&et=heartbeat'
        response = requests.get(url, self.Params('stalled', video_id, 0, 0, 0, sq), headers=self.cookie, verify=False)
        print('发送stalled包')
        return True

    def Seeking(self, video_id, last, total, sq):
        url = 'https://scutspoc.xuetangx.com/heartbeat?i=5&et=heartbeat'
        response = requests.get(url, self.Params('seeking', video_id, last, total, last, sq), cookies=self.cookie,
                                verify=False)
        print('发送seeking包')
        return True

    def Loadeddata(self, video_id, last, total, sq):

        url = 'https://scutspoc.xuetangx.com/heartbeat?i=5&et=heartbeat'
        response = requests.get(url, self.Params('loadeddata', video_id, 0, total, last, sq), cookies=self.cookie,
                                verify=False)
        print('发送loaddata包')
        return True

    def play(self, video_id, last, total, sq, cp=0.0):
        url = 'https://scutspoc.xuetangx.com/heartbeat?i=5&et=heartbeat'
        response = requests.get(url, self.Params('play', video_id, cp, total, last, sq), cookies=self.cookie,
                                verify=False)
        print('发送play包')
        return True

    def waiting(self, video_id, last, total, sq):
        url = 'https://scutspoc.xuetangx.com/heartbeat?i=5&et=heartbeat'
        response = requests.get(url, self.Params('waiting', video_id, 0.9, total, last, sq), cookies=self.cookie,
                                verify=False)
        print('发送waiting包')
        return True

    def playing(self, video_id, last, total, sq, cp=0.0):
        url = 'https://scutspoc.xuetangx.com/heartbeat?i=5&et=heartbeat'
        response = requests.get(url, self.Params('playing', video_id, cp, total, last, sq), cookies=self.cookie,
                                verify=False)
        print('发送playing包')
        return True

    def heartbeat(self, video_id, cp, last, total, sq):
        url = 'https://scutspoc.xuetangx.com/heartbeat?i=5&et=heartbeat'
        response = requests.get(url, self.Params('playing', video_id, cp, total, last, sq), cookies=self.cookie,
                                verify=False)
        print('发送heartbeat包')
        return True

    def Pause(self, video_id, cp, last, total, sq):
        url = 'https://scutspoc.xuetangx.com/heartbeat?i=5&et=heartbeat'
        response = requests.get(url, self.Params('pause', video_id, cp, total, last, sq), cookies=self.cookie,
                                verify=False)
        print('间隔性发送pause包')
        return True

    def videoend(self, video, last, total, sq):
        url = 'https://scutspoc.xuetangx.com/heartbeat?i=5&et=heartbeat'
        response = requests.get(url, self.Params('videoend', video, total, total, last, sq), cookies=self.cookie,
                                verify=False)
        print('发送videoend包')
        return True

    def Params(self, method, video_id, cp, d, tp, sq):
        millis = int(round(time.time() * 1000))
        params = {
            'i': 5,
            'et': method,
            'p': 'web',
            'n': self.getVideoGroup(),
            'lob': 'cloud3',
            'cp': cp,
            'fp': 0,
            'tp': tp,
            'sp': 1,
            'ts': millis,
            'u': self.getUserInfo()['user_id'],
            'c': self.getCourseId(),
            'v': video_id,
            'cc': video_id,
            'd': d,
            'pg': str(video_id) + '_qak3',
            'sq': sq,
            't': 'video',
        }
        return params

    def CheckCookie(self):
        url = 'https://scutspoc.xuetangx.com/api/forum/v1/user/data/?user_id=' + str(self.getUserInfo()['user_id'])
        try:
            response = requests.get(url, timeout=5, cookies=self.cookie)
            content = response.json()
            if content['message'] == 'unauthorized':
                return False
            else:
                return True
        except:
            return True

    def GetLong(self, video_url):
        cap = cv2.VideoCapture(video_url)
        # file_path是文件的绝对路径，防止路径中含有中文时报错，需要解码
        if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
            # get方法参数按顺序对应下表（从0开始编号)
            rate = cap.get(5)  # 帧速率
            FrameNumber = cap.get(7)  # 视频文件的帧数
            duration = FrameNumber / rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
            time = math.ceil(duration * 10)
            return time / 10
        else:
            return 0

    def GetCode(self):
        response = requests.get('https://scutspoc.xuetangx.com/api/v1/code/captcha', timeout=10,
                                verify=False)
        content = json.loads(response.text)['data']
        img = content['img']
        captcha_key = content['captcha_key']

        img_data = base64.b64decode(img)
        nparr = np.fromstring(img_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return [captcha_key, img_np]

    def InputInfo(self):
        self.username = input('Please input your username:')
        self.password = input('Please input your password:')

        pass

    def MockLogin(self, cnn):
        try_number = 3
        status = False
        np_train_datas = np.empty(shape=(4, 50, 50, 1), dtype='float32')
        while try_number > 0 and status is False:
            key, code_img = self.GetCode()
            img = QuZao(code_img)
            list = FenGe(img)
            fenge = DrawLine(img, list)
            if fenge != []:
                img_list = [cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (50, 50), interpolation=cv2.INTER_CUBIC) for
                            i in fenge]
                for i in range(4):
                    np_train_datas[i] = img_list[i][:, :, np.newaxis] / 256  # 归一化g

                predicts = cnn.session.run(cnn.prediction, feed_dict={cnn.x_data: np_train_datas})
                predict = [np.unravel_index(predict.argmax(), predict.shape)[0] for predict in predicts]
            else:
                print('验证码识别出错，自动重新识别')
                try_number -= 1
                continue
            code = ''.join([Int2Chart(i) for i in predict])
            data = {'captcha': code, 'captcha_key': key, 'is_alliance': 0, 'login': self.username,
                    'password': self.password}
            try:
                response = requests.post('https://scutspoc.xuetangx.com/api/v1/oauth/number/login', data=data,
                                         timeout=5)
                if response.status_code != 200:
                    print('登陆出错! Response: ', response.text)
                    if '图形验证码错误' in response.text:
                        print('判断为验证码识别出错，自动重新识别')
                        try_number -= 1
                        continue
                    else:
                        break
                self.cookie.update(response.cookies)
                status = True
            except:
                print('登陆超时，自动重试')
                try_number -= 1
                continue
        return status

    def GetCourseList(self):
        status = True
        number = 1
        courseList = []
        while status:
            url = 'https://scutspoc.xuetangx.com/mycourse_list?running_status=&term_id=&search=&page_size=10&page=' + str(
                number)
            try:
                response = requests.get(url, cookies=self.cookie, verify=False, timeout=10)
                courseList += response.json()['data']['results']
                if len(courseList) >= response.json()['data']['count']:
                    status = False
            except:
                result = input('获取课程列表超时,选择重试还是退出?[Y/N]:')
                if result == 'Y' or result == 'y':
                    continue
                else:
                    break
        if status == True:
            return False
        courses = {}
        for key, course in enumerate(courseList):
            if course['status'] != 'ing':
                continue
            print('【', str(key), '】', course['course_name'], ' | ', course['term_name'], ' | ', course['class_name'],
                  ' | ', '课程ID:',
                  course['course_id'], ' | ', course['status'])
            courses.update({str(key): course['course_id']})
        return courses


def main():
    print('正在初始化验证码识别模型')
    Test = test()
    cnn = cnn_net.cnn_net()
    cnn.saver.restore(cnn.session, cnn.save_path)
    print('初始化结束')
    Test.InputInfo()
    status = Test.MockLogin(cnn)
    if status is False:
        print('登陆失败!')
        return
    else:
        print('登陆成功!')
    print('正在上课的课程列表:')
    courses = Test.GetCourseList()
    status = True
    while status:
        courses_id = input('请输入课程前的序号，进入课程: ')
        if str(courses_id) in courses:
            print('进入课程ID为：', courses[courses_id])
            Test.course = str(courses[courses_id])
            status = False
        else:
            print('没有序号为', courses_id, '的课程')

    threads = []
    Videos = Test.loadRecordToVideos()

    for video in Videos:
        video_id = video['item_id']
        video_name = video['section']
        try:
            last = video['last']
        except:
            last = 0
        t = threading.Thread(target=Test.proccess, args=(video_id, last, video_name))
        threads.append(t)
    for i in threads:
        i.start()
    for a in threads:
        a.join()

    print('已播完所有视频')


if __name__ == '__main__':
    main()