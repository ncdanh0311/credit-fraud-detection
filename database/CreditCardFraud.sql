-- =====================================================================
-- SCRIPT KHỞI TẠO CƠ SỞ DỮ LIỆU & BẢNG GHI VẾT GIAO DỊCH (SQL SERVER)
-- =====================================================================

-- 1. Tạo Cơ sở dữ liệu CreditCardFraudDB (Nếu chưa tồn tại)
IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'CreditCardFraudDB')
BEGIN
    CREATE DATABASE CreditCardFraudDB;
    PRINT 'Da tao thanh cong co so du lieu CreditCardFraudDB.';
END
ELSE
BEGIN
    PRINT 'Co so du lieu CreditCardFraudDB da ton tai.';
END
GO

-- Sử dụng cơ sở dữ liệu vừa tạo
USE CreditCardFraudDB;
GO

-- 2. Tạo Bảng TransactionLogs ghi vết nhật ký kiểm thử giao dịch
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[TransactionLogs]') AND type in (N'U'))
BEGIN
    CREATE TABLE [dbo].[TransactionLogs] (
        [LogID] INT IDENTITY(1,1) NOT NULL,
        [TransactionID] VARCHAR(50) NOT NULL,
        [ModelName] NVARCHAR(100) NOT NULL,
        [Prediction] VARCHAR(20) NOT NULL,
        [Probability] FLOAT NOT NULL,
        [ExecutionTimeMs] INT NOT NULL,
        [InputTime] FLOAT NOT NULL,
        [InputAmount] FLOAT NOT NULL,
        [V_Values] VARCHAR(MAX) NOT NULL,
        [Timestamp] DATETIME CONSTRAINT [DF_TransactionLogs_Timestamp] DEFAULT (GETDATE()) NOT NULL,
        
        -- Khóa chính
        CONSTRAINT [PK_TransactionLogs] PRIMARY KEY CLUSTERED ([LogID] ASC)
    );
    
    PRINT 'Da tao thanh cong bang TransactionLogs.';
    
    -- Thêm một số dữ liệu mẫu mô phỏng ban đầu để Dashboard có sẵn thông tin hiển thị
    INSERT INTO [dbo].[TransactionLogs] ([TransactionID], [ModelName], [Prediction], [Probability], [ExecutionTimeMs], [InputTime], [InputAmount], [V_Values], [Timestamp])
    VALUES 
    ('TXN-005', N'1D-CNN - SMOTE', 'NORMAL', 0.0115, 68, 1205.0, 15.00, '-1.23,0.45,-0.89,0.12,0.56,-0.34,0.12,-0.05,0.78,-0.12,0.01,0.23,-0.45,0.67,-0.89,0.12,-0.23,0.45,-0.12,0.02,0.12,-0.04,0.05,-0.12,0.23,-0.01,0.02,-0.05', DATEADD(minute, -30, GETDATE())),
    ('TXN-006', N'Random Forest - NoSMOTE', 'NORMAL', 0.0421, 45, 3600.0, 150.00, '0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0', DATEADD(minute, -15, GETDATE()));
    
    PRINT 'Da nap du lieu mau thanh cong.';
END
ELSE
BEGIN
    PRINT 'Bang TransactionLogs da ton tai.';
END
GO
