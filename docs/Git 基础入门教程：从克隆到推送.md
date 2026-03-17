# Git 基础入门教程：从克隆到推送

本教程面向从未使用过 Git 的初学者，将指导你如何安装 Git、配置环境、克隆 GitHub 仓库，以及如何使用 `pull` 和 `push` 命令进行代码同步。

[toc]

## 1. Git 简介与安装

Git 是一个分布式版本控制系统，它可以记录文件的修改历史，方便多人协作开发。

**安装步骤：**

- **Windows:** 访问 [Git 官网](https://git-scm.com/) 下载安装包，一路点击 “Next” 完成安装。
- **Mac:** 打开终端，输入 `git --version`，系统会自动提示安装 Xcode Command Line Tools（包含 Git）。
- **Linux (Debian/Ubuntu):**

```bash
sudo apt-get install git
```

**验证安装：**
打开终端或 Git Bash，输入：

```bash
git --version
```

若显示版本号，则安装成功。

## 2. 初始配置

在第一次使用 Git 前，你需要告诉 Git 你是谁（你的名字和邮箱），这很重要，因为每次提交都会记录这些信息。

打开终端，输入以下命令：

```bash
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub注册邮箱"
```

> **注意：** `--global` 表示全局配置，意味着这台电脑上所有的 Git 项目都会默认使用这个身份。

## 3. 克隆仓库

如果你在 GitHub 上看到一个喜欢的项目，或者需要参与团队开发，第一步就是把它“下载”到本地，在 Git 术语中称为“克隆”。

**步骤：**

1. 在 GitHub 仓库页面点击绿色的 **Code** 按钮，复制链接（通常选择 HTTPS）。
2. 在终端中进入你想存放项目的文件夹。
3. 执行 `git clone` 命令。

**示例：**

```bash
# 进入桌面目录(或你想要存放项目的位置，将Desktop更换为指定目录即可)
cd Desktop

# 克隆仓库（将下面的链接替换为你的实际链接）
git clone https://github.com/用户名/项目名.git
```
执行后，桌面上会出现一个以项目名命名的文件夹。

## 4. 核心工作流

在执行 `push` 之前，你需要理解 Git 的三个区域：

1. **工作区:** 你肉眼看到的文件，正在编辑的代码。
2. **暂存区:** 准备提交的修改（通过 `git add` 放入）。
3. **版本库:** 已经提交的历史记录（通过 `git commit` 确认）。

**操作流程：**

首先，进入刚才克隆下来的项目文件夹：

```bash
cd 项目名
```
### 步骤 A: 查看状态

随时可以使用此命令查看当前状态：

```bash
git status
```
### 步骤 B: 添加到暂存区

假设你修改了 `README.md` 文件。你需要告诉 Git 把这个修改加入“待提交清单”。

```bash
# 添加指定文件
git add README.md

# 或者添加所有修改过的文件（最常用）
git add .
```

### 步骤 C: 提交更改

将暂存区的内容正式保存到版本库，并附上说明。
```bash
git commit -m "这里写本次修改的说明，例如：修改了README文件"
```

## 5. 上传代码

当你完成了本地的 `commit`，代码只存在于你的电脑上。要把它传送到 GitHub，需要使用 `push`。

**命令：**
```bash
# 将本地提交推送到远程仓库
git push
```

**第一次 Push 注意事项：**
如果你是第一次推送该分支，Git 可能会提示你建立关联，按照提示输入：
```bash
git push --set-upstream origin main
```

*(注：新版 Git 默认主分支名为 `main`，旧版可能是 `master`)*

**身份验证：**
执行 `push` 时，终端可能会弹出登录窗口或要求输入密码。

- **用户名:** 输入你的 GitHub 用户名。
- **密码:** **不要输入登录密码**，你需要输入 GitHub 的 **Personal Access Token (PAT)**。
  - *获取方式：GitHub网页 -> Settings -> Developer settings -> Personal access tokens -> Generate new token。*

## 6. 下载更新

当团队其他人更新了 GitHub 上的代码，或者你在另一台电脑上修改了代码，你需要将这些更新同步到本地。

**命令：**
```bash
git pull
```

这个命令会将远程仓库的最新代码下载下来并与你的本地代码合并。

> **最佳实践：** 在每天开始工作或准备 `push` 之前，先执行一次 `git pull`，以减少冲突。

## 7. 常见问题与总结

### 常用命令速查表

| 命令                   | 作用                   |
| :--------------------- | :--------------------- |
| `git clone <url>`      | 克隆远程仓库到本地     |
| `git status`           | 查看当前文件状态       |
| `git add .`            | 将所有修改放入暂存区   |
| `git commit -m "信息"` | 提交暂存区到本地仓库   |
| `git push`             | 推送本地提交到远程仓库 |
| `git pull`             | 拉取远程更新到本地     |

### 遇到冲突怎么办？

如果在 `pull` 或 `push` 时提示 `CONFLICT`，说明你和别人修改了同一个文件的同一行。Git 无法自动判断保留哪一个。

1. 打开冲突的文件，你会看到类似 `<<<<<<<` 和 `>>>>>>>` 的标记。
2. 手动编辑文件，保留正确的代码，删除标记符号。
3. 重新执行 `git add .` 和 `git commit -m "解决冲突"`。
4. 最后执行 `git push`。