---
title: 'github pages + Hexo + 域名绑定'
date: 2016-12-19 23:03:28
---
本文从安装环境开始详细讲述如何使用github pages + Hexo搭建自己的静态博客，并且使用阿里云绑定域名。
<!--more-->
**环境**
1. 安装Git![](http://oifuxc6w5.bkt.clouddn.com/git.png)
2. 安装Node![](http://oifuxc6w5.bkt.clouddn.com/NODEJS.png)
3. 验证安装![](http://oifuxc6w5.bkt.clouddn.com/cmd.png)

**Github Pages**
 在GitHub创建一个格式为：yourusername.github.io的仓库即可。
![](http://oifuxc6w5.bkt.clouddn.com/gitmaven.png)
**Hexo**
Hexo 是一个快速、简洁且高效的博客框架。Hexo 使用 Markdown（或其他渲染引擎）解析文章，在几秒内，即可利用靓丽的主题生成静态网页。
建立一个blog文件夹用于存放博客文件,然后右键单击选择“Git Bash”。
`npm install hexo-cli -g`
`hexo init blog`
`cd blog`
`npm install`
`hexo server`
执行hexo server时，默认端口是4000,如果端口被占用更换即可。
如`hexo server -p 4001`
访问`http://localhost:4000/`
![](http://oifuxc6w5.bkt.clouddn.com/hexo.png)
**更换主题**
exo-theme：https://hexo.io/themes/
hexo-github-theme-list：https://github.com/hexojs/hexo/wiki/Themes
`git clone`加上主题地址
_config.yml中将theme改成刚刚下载的主题
修改完成后`hexo generate`  `hexo server`重启服务器查看效果
![](http://oifuxc6w5.bkt.clouddn.com/newtheme.png)
**部署代码到github**
``ssh-keygen -t rsa -C "你的邮箱地址"``生成SSH秘钥
在https://github.com/settings/ssh 添加id_rsa.pub中的秘钥
安装插件：
`npm install hexo -server --save`
`npm install hexo-deployer-git --save`
安装其他插件的格式为`npm install ... --save`
编辑全局 hexo 的配置文件：`_config.yml`
注意`:`后留一个空格
```deploy: 
  type: git
  repository: https://github.com/BalianCheng/BalianCheng.github.io.git
  branch: master```
编辑全局配置后需要重新部署：
清除掉已经生成的文件：`hexo clean`
再生成静态文件：`hexo generate`
预览：`hexo server`
打开`localhost:4000`查看
部署：`hexo deploy`
生成 40 4页面：`hexo new page 404`
生成 about 页面：`hexo new page about`
生成 tag 标签云页面：`hexo new page tags`
**绑定域名**
`ping yourname.github.io`获得IP地址并使用域名解析
进入GitHub项目,进入`Settings`,在`Custom domain`写入域名
在\blog\public下建立CNAME文件写入域名
