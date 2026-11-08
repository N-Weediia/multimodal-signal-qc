# 2026-09-14 预处理复核
本轮从 data/ 重新扫描 7 位受试者，共 107 条记录；新增 sec_test（中评/第二测试）数据已自动纳入。复跑结果：READY_WITH_MASKS 41、READY_CHANNEL_FILTERED 33、READY_CONTINUOUS 14、READY_WITH_MASKS_KNOWN_IR_INTERFERENCE 16、FAILED 3，与既有索引一致，说明新增数据识别稳定且未覆盖原始文件。

处理前后：EEG 1000→250 Hz、50 Hz 陷波、0.5–45 Hz 带通、坏导联与坏段 mask；fNIRS 非正/缺失样本剔除、突变修复、0.01–0.2 Hz 滤波，保留 OD/相对 MBLL；EMG 20–450 Hz 带通、50 Hz 谐波抑制并保留包络。

改进建议：按受试者×访视×任务去重；重复导出文件先人工确认；3 条 EEG 直接信号失败继续排除，不生成伪恢复数据；后续统计继续使用质量 mask。
