# Contributing to rga_gpu_cpu_stitch

感谢你的关注！欢迎通过 Issue 与 Pull Request 参与本项目。

Thank you for your interest! Contributions via issues and pull requests are welcome.

## 如何贡献 / How to contribute

1. **报告问题 / Report an issue**：描述环境（RK3588 板卡、系统、内核、OpenCV/librga/rknnrt 版本）、复现步骤与期望/实际行为。
   Describe the environment (RK3588 board, OS, kernel, OpenCV/librga/rknnrt versions), steps to reproduce, and expected vs. actual behavior.
2. **提交代码 / Submit a PR**：
   - Fork 仓库并新建分支（命名如 `fix/xxx`、`feat/xxx`）。
   - 保持编码风格与现有代码一致（C++17、4 空格缩进、`-Wall -Wextra` 无警告）。
   - 提交信息简洁清晰，说明动机与改动。
   - Fork the repo, create a branch (e.g. `fix/xxx`, `feat/xxx`).
   - Keep style consistent with existing code (C++17, 4-space indent, no warnings under `-Wall -Wextra`).
   - Write concise commit messages describing motivation and changes.

## 编码规范 / Code style

- C++17，源文件使用 UTF-8 编码（无 BOM）。
- 头文件使用 `#pragma once`。
- 日志/打印使用 `printf`，与现有代码保持一致。
- C++17; source files are UTF-8 (no BOM).
- Headers use `#pragma once`.
- Logging uses `printf` to match existing code.

## 敏感信息 / Sensitive data

请勿提交任何 IP 地址、密码、密钥、私有路径或未脱敏的品牌信息。

Do not commit IP addresses, passwords, secrets, private paths, or non-sanitized branding.

## 许可 / License

贡献默认以本项目 [MIT](./LICENSE) 许可证授权。

By contributing you agree to license your work under the project's [MIT](./LICENSE) license.
