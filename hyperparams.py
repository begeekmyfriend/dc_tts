# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''
class Hyperparams:
    '''Hyper parameters'''
    # pipeline
    prepro = True  # if True, run `python prepro.py` first before running `python train.py`.
    
    # signal processing
    sr = 16000  # Sampling rate.
    n_fft = 1024  # fft points (samples)
    n_mgc = 60
    n_lf0 = 1
    n_vuv = 1
    n_bap = 1
    use_harvest = True
    frame_period = 15  # seconds
    frame_length = 0.06  # seconds

    # Model
    r = 4 # Reduction factor. Do not change this.
    dropout_rate = 0.05
    e = 128 # == embedding
    d = 256 # == hidden units of Text2Mel
    c = 512 # == hidden units of SSRN
    attention_win_size = 3

    # data
    vocab = "_~ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz12345',.?!:;@" # _: Padding, ~: EOS.
    max_N = 300 # Maximum number of characters.
    max_T = 700 # Maximum number of mel frames.

    # training scheme
    lr = 0.001 # Initial learning rate.
    logdir = "logdir/mandarin"
    sampledir = 'samples'
    B = 48 # batch size
    num_iterations = 2000000

    # test_data
    sentences = [
        "huan2 qiu2 wang3 bao4 dao4",
        "e2 luo2 si1 wei4 xing1 wang3 shi2 yi1 ri4 bao4 dao4 cheng1",
        "ji4 nian4 di4 yi2 ci4 shi4 jie4 da4 zhan4 jie2 shu4 yi4 bai3 zhou1 nian2 qing4 zhu4 dian3 li3 zai4 ba1 li2 ju3 xing2",
        "e2 luo2 si1 zong3 tong3 pu3 jing1 he2 mei3 guo2 zong3 tong3 te4 lang3 pu3 zai4 ba1 li2 kai3 xuan2 men2 jian4 mian4 shi2 wo4 shou3 zhi4 yi4",
        "pu3 jing1 biao3 shi4",
        "tong2 mei3 guo2 zong3 tong3 te4 lang3 pu3 jin4 xing2 le hen3 hao3 de jiao1 liu2",
        "e2 luo2 si1 zong3 tong3 zhu4 shou3 you2 li3",
        "wu1 sha1 ke1 fu1 biao3 shi4",
        "fa3 guo2 fang1 mian4 zhi2 yi4 yao1 qiu2 bu2 yao4 zai4 ba1 li2 ju3 xing2 ji4 nian4 huo2 dong4 qi1 jian1 ju3 xing2 e2 mei3 liang3 guo2 zong3 tong3 de dan1 du2 hui4 wu4",
        "da2 cheng2 le xie2 yi4",
        "wo3 men yi3 jing1 kai1 shi3 xie2 tiao2 e2 luo2 si1 he2 mei3 guo2 zong3 tong3 hui4 wu4 de shi2 jian1",
        "dan4 hou4 lai2 wo3 men kao3 lv4 dao4 le fa3 guo2 tong2 hang2 men de dan1 you1 he2 guan1 qie4",
        "wu1 sha1 ke1 fu1 shuo1",
        "yin1 ci3",
        "wo3 men yu3 mei3 guo2 dai4 biao3 men yi4 qi3 jin4 xing2 le tao3 lun4",
        "jue2 ding4 zai4 bu4 yi2 nuo4 si1 ai4 li4 si1 feng1 hui4 shang4 jin4 xing2 nei4 rong2 geng4 feng1 fu4 de dui4 hua4",
        "bao4 dao4 cheng1",
        "pu3 jing1 he2 te4 lang3 pu3 zai4 ai4 li4 she4 gong1 wu3 can1 hui4 shang4 de zuo4 wei4 an1 pai2 zai4 zui4 hou4 yi1 fen1 zhong1 jin4 xing2 le tiao2 zheng3",
        "dan4 zhe4 bing4 bu4 fang2 ai4 ta1 men jiao1 tan2",
        "sui1 ran2 dong1 dao4 zhu3 fa3 guo2 dui4 ta1 men zai4 ba1 li2 de hui4 wu4 biao3 shi4 fan3 dui4",
        "dan4 e2 mei3 ling3 dao3 ren2 reng2 ran2 biao3 shi4",
        "ta1 men xi1 wang4 zai4 ai4 li4 she4 gong1 de gong1 zuo4 wu3 can1 shang4 hui4 mian4",
        "chu1 bu4 zuo4 wei4 biao3 xian3 shi4",
        "te4 lang3 pu3 bei4 an1 pai2 zai4 pu3 jing1 pang2 bian1",
        "dan4 zai4 sui2 hou4 jin4 xing2 de gong1 zuo4 wu3 can1 qi1 jian1",
        "zuo4 wei4 an1 pai2 xian3 ran2 yi3 jing1 fa1 sheng1 le bian4 hua4",
        "cong2 zhao4 pian1 lai2 kan4",
        "pu3 jing1 dang1 shi2 zheng4 quan2 shen2 guan4 zhu4 de yu3 lian2 he2 guo2 mi4 shu1 chang2 gu3 te4 lei2 si1 jiao1 tan2",
        "ou1 meng2 wei3 yuan2 hui4 zhu3 xi2 rong2 ke4 zuo4 zai4 pu3 jing1 de you4 bian1",
        "er2 te4 lang3 pu3 ze2 zuo4 zai4 ma3 ke4 long2 pang2 bian1",
        "ma3 ke4 long2 de you4 bian1 ze2 shi4 de2 guo2 zong3 li3 mo4 ke4 er3",
        "ci3 qian2",
        "pu3 jing1 zai4 fang3 wen4 ba1 li2 qi1 jian1 biao3 shi4",
        "ta1 bu4 pai2 chu2 yu3 te4 lang3 pu3 zai4 gong1 zuo4 wu3 can1 shi2 jin4 xing2 jiao1 liu2",
        "pu3 jing1 zai4 fa3 guo2 pin2 dao4 de jie2 mu4 zhong1 hui2 da2 shi4 fou3 yi3 tong2 te4 lang3 pu3 jin4 xing2 jiao1 liu2 de wen4 ti2 shi2 biao3 shi4 zan4 shi2 mei2 you3",
        "wo3 men zhi3 da3 le ge4 zhao1 hu1",
        "yi2 shi4 yi3 zhe4 yang4 de fang1 shi4 jin4 xing2",
        "wo3 men wu2 fa3 zai4 na4 li3 jin4 xing2 jiao1 liu2",
        "wo3 men guan1 kan4 le fa1 sheng1 de shi4 qing2",
        "dan4 xian4 zai4 hui4 you3 gong1 zuo4 wu3 can1",
        "ye3 xu3 zai4 na4 li3 wo3 men hui4 jin4 xing2 jie1 chu4",
        "dan4 shi4",
        "wu2 lun4 ru2 he2",
        "wo3 men shang1 ding4",
        "wo3 men zai4 zhe4 li3 bu4 hui4 wei2 fan3 zhu3 ban4 guo2 de gong1 zuo4 an1 pai2",
        "gen1 ju4 ta1 men de yao1 qiu2",
        "wo3 men bu4 hui4 zai4 zhe4 li3 zu3 zhi1 ren4 he2 hui4 mian4",
        "er2 shi4 ke3 neng2 hui4 zai4 G er4 ling2 qi1 jian1 huo4 zai4 ci3 zhi1 hou4 ju3 xing2 hui4 mian4",
        "pu3 jing1 hai2 biao3 shi4",
        "e2 luo2 si1 zhun3 bei4 tong2 mei3 guo2 jin4 xing2 dui4 hua4",
        "fan3 zheng4 bu2 shi4 mo4 si1 ke1 yao4 tui4 chu1 zhong1 dao3 tiao2 yue1",
    ]
