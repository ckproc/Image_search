生成时间：11月25日

SDK准确率： 95%左右

SDK的配置参数在config.cfg中，
——retrievalType：默认为 STATIC
——dictDim1：码本1大小，建议值为 1200
——dictDim2：码本2大小，建议值为 50
——usePCA：建议值为 true
——pcaDim：PCA降维维度，建议值为 50
——encodeType：默认为 FV
——basePath：码本生成路径
——databasePath：数据库文本

改动：
替换kdtree，去掉retrievalImage函数内部的互斥锁，使检索支持多线程。

注意: C++接口请参照test.py中的getSimpleResult函数。


ps: 生成字典时，默认直接使用数据库中的目标图片。
当有新的目标图片需要识别时，无需立即更新字典。