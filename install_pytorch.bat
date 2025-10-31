@echo off
REM 安装 PyTorch（按照README要求：torch==1.12.1, torchvision==0.13.1）
REM 注意：由于 PyTorch 1.12.1 已不在官方 pip 仓库，改用 conda 安装（已在 environment.yml 中配置）
REM 如果环境已创建，可以运行以下命令手动安装：
echo ========================================
echo PyTorch 安装提示
echo ========================================
echo.
echo 由于 PyTorch 1.12.1 已不在官方 pip 仓库，请使用 conda 安装：
echo.
echo    conda install pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.6 -c pytorch
echo.
echo 或者在创建环境时，environment.yml 已经包含了这些依赖，直接运行：
echo    conda env update -n adapt-cmvc -f environment.yml --prune
echo.
echo ========================================
pause

