开始项目

    第一次启动，请参考requirement.txt安装

如何训练模型？

    从命令行开始 python train_console.py
    
        可选参数：
            --launch_mode    启动模式：默认train（目前仅train）
            --dataset_dir    训练数据目录
            --valid_dir      验证数据目录
            --test_dir       测试数据目录
            --net_path       网络保存路径
            --anchors        参考框（注：选填示例 --anchors "{\"13\": [[1, 2]]}"）
            --epochs         训练轮次
            --batch_size     训练批次大小
            --is_new         是否重头开始训练网络（True or False）
            --log_dir        loss画图的保存地址
            --plot_interval  loss画图的间隔轮次
            --save_loss_plot 是否保存loss画图（True or False）
            --plot_pause     画图的停留时间（默认为0.001，单位为秒）
            --plot_loss      是否loss画图（True or False）
            --optimizer      指定优化器，默认为Adam（可选SGD）

如何使用训练结果

    1. 可运行目录下的文件‘eval.py’
    
        python eval.py --path img_file_path --net_path(optional) net_file_path
        
        img_file_path：图片文件路径
        net_file_path：可指定网络文件路径，默认读取同目录下‘./model/yolo-tiny.pth’
    
    2.如果需要在另外的.py文件中调用，调用eval.py中的Eval函数即可，需要以下参数
    
        path：必须参数，为图片文件位置
        net_path：可选参数，为网络文件位置
        img: 可选参数，可传图片数据，如果该参数有效则path可任意填写
        
    3. 关于返回数据
        
        图片上的目标位置
        ndarray [[IOU, X, Y, W, H]]
        
        分类结果
        ndarray [[1. 0. 0. 0. 0.]]
