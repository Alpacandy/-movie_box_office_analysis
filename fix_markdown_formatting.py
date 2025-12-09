#!/usr/bin/env python3
"""
Markdown格式修复脚本
用于修复operation_manual.md中的Markdown格式问题
包括：
1. 为代码块添加语言指定
2. 修复裸URL为Markdown链接格式
3. 添加标题周围的空行
4. 添加列表周围的空行
5. 添加代码块周围的空行
"""

import re
import os

def fix_markdown_formatting(file_path):
    """修复Markdown文件格式"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 为代码块添加语言指定（主要是sh和txt语言）
    # 匹配没有语言指定的代码块，根据内容判断应该使用的语言
    def add_language_spec(match):
        code_block = match.group(0)
        # 处理代码块标记后有一个空行的情况
        if code_block.startswith('```\n\n'):
            code_block = code_block.replace('```\n\n', '```\n')
        # 如果代码块内容包含命令或命令输出，使用sh语言
        if re.search(r'(python|pip|kaggle|chmod|ERROR:|INFO:)', code_block):
            return re.sub(r'^```$', '```sh', code_block, count=1, flags=re.MULTILINE)
        # 否则使用txt语言
        else:
            return re.sub(r'^```$', '```txt', code_block, count=1, flags=re.MULTILINE)
    
    # 匹配没有语言指定的代码块
    content = re.sub(r'```\n(?:(?!```).)*?```', add_language_spec, content, flags=re.DOTALL)
    
    # 2. 先修复已存在的重复嵌套链接（清理之前错误导致的问题）
    # 匹配格式为 [[text](url)](url) 的嵌套链接
    while True:
        new_content = re.sub(r'\[\[(.*?)\]\((.*?)\)\]\(\2\)', r'[\1](\2)', content)
        if new_content == content:
            break
        content = new_content
    
    # 3. 修复裸URL为Markdown链接格式
    # 使用更精确的URL匹配模式，避免匹配到已在Markdown链接中的URL
    url_pattern = r'(?<!\[)(?<!\()https?://[^\s<\)]+'
    # 使用迭代方式替换，避免并发修改问题
    # 先找到所有URL位置
    urls = list(re.finditer(url_pattern, content))
    # 从后往前替换，避免位置偏移
    for url_match in reversed(urls):
        url = url_match.group(0)
        start = url_match.start()
        end = url_match.end()
        # 检查URL是否已经在Markdown链接中
        context = content[max(0, start-20):end+20]
        # 修复正则表达式语法，使用原始字符串
        link_pattern = r'\[.*?\]\(.*?' + re.escape(url) + r'.*?\)'
        if not re.search(link_pattern, context):
            # 替换为Markdown链接格式
            content = content[:start] + f'[{url}]({url})' + content[end:]
    
    # 4. 添加标题周围的空行（MD022）
    # 分两步处理：先处理标题前面的空行，再处理标题后面的空行
    
    # 匹配标题前面没有空行的情况
    content = re.sub(r'(?<!^)(?<!\n\n)(\n)(#{1,6} )', r'\n\n\2', content)
    
    # 匹配标题后面没有空行的情况
    content = re.sub(r'(#{1,6} .*)(\n)(?!\n\n)(?!^#)', r'\1\n\n', content)
    
    # 5. 添加列表周围的空行（MD032）
    # 匹配无序列表项（- 或 * 开头）
    content = re.sub(r'(?<!^)(?<!\n\n)(\n)([\-\*] )', r'\n\n\2', content)
    
    # 匹配有序列表项（数字. 开头）
    content = re.sub(r'(?<!^)(?<!\n\n)(\n)(\d+\. )', r'\n\n\2', content)
    
    # 匹配列表结束（后面不是空行，且下一行不是列表项）
    content = re.sub(r'([\-\*] .*)(\n)(?!\n\n)(?!^[\-\*] )(?!^\d+\. )', r'\1\n\n', content, flags=re.MULTILINE)
    
    # 6. 添加代码块周围的空行（MD031）
    # 匹配代码块开始标记（前面不是空行）
    content = re.sub(r'(?<!^)(?<!\n\n)(\n)(```)', r'\n\n\2', content)
    
    # 匹配代码块结束标记（后面不是空行）
    content = re.sub(r'(```)(\n)(?!\n\n)(?!^```)', r'\1\n\n', content, flags=re.MULTILINE)
    
    # 7. 确保文件开头没有多余空行
    content = content.lstrip('\n')
    
    # 8. 确保文件结尾有一个空行
    if not content.endswith('\n'):
        content = content + '\n'
    
    # 9. 修复连续的空行（最多保留1个，解决MD012）
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 11. 修复邮箱地址格式（将邮箱转换为链接格式，解决MD034对邮箱的误判）
    email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    content = re.sub(email_pattern, r'[\1](mailto:\1)', content)
    
    # 12. 修复文件末尾的空行（确保文件末尾只有0或1个空行）
    content = content.rstrip('\n') + '\n'
    
    # 10. 修复表格格式问题（MD060） - 这里只做简单处理，将表格管道符周围添加空格
    # 匹配表格行（包含|的行）
    def fix_table_format(line):
        if '|' in line and not line.strip().startswith('---'):
            # 在|周围添加空格
            line = re.sub(r'\|', ' | ', line)
            # 移除多余的空格
            line = re.sub(r'\s{2,}', ' ', line)
            # 确保开头和结尾的格式正确
            line = line.strip()
            if not line.startswith('|'):
                line = '| ' + line
            if not line.endswith('|'):
                line = line + ' |'
        return line
    
    # 逐行处理文件内容
    lines = content.split('\n')
    fixed_lines = [fix_table_format(line) for line in lines]
    content = '\n'.join(fixed_lines)
    
    return content

def main():
    """主函数"""
    file_path = os.path.join(os.path.dirname(__file__), 'operation_manual.md')
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    print(f"正在修复文件: {file_path}")
    
    fixed_content = fix_markdown_formatting(file_path)
    
    # 保存修复后的内容
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Markdown格式修复完成！")

if __name__ == "__main__":
    main()
