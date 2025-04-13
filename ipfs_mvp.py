import requests
import json
import os
import argparse
import time

class SimpleIPFSConnector:
    """簡單的IPFS連接器，使用requests庫與IPFS API通信"""
    
    def __init__(self, api_url="http://localhost:5001/api/v0"):
        """初始化IPFS連接器
        
        Args:
            api_url: IPFS API端點，默認為本地節點
        """
        self.api_url = api_url
        # 測試連接
        try:
            response = requests.post(f"{self.api_url}/id")
            if response.status_code == 200:
                node_id = response.json()["ID"]
                print(f"成功連接到IPFS節點: {node_id}")
                print(f"節點地址: {response.json()['Addresses']}")
            else:
                print(f"無法連接到IPFS API: {response.status_code}")
        except Exception as e:
            print(f"連接IPFS時出錯: {str(e)}")
    
    def add_file(self, file_path):
        """將檔案添加到IPFS
        
        Args:
            file_path: 要添加的檔案路徑
            
        Returns:
            包含CID的字典
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"檔案不存在: {file_path}")
            
        print(f"正在添加檔案到IPFS: {file_path}")
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            response = requests.post(f"{self.api_url}/add", files=files)
            
        if response.status_code != 200:
            raise Exception(f"IPFS add請求失敗: {response.text}")
            
        result = response.json()
        print(f"檔案已添加到IPFS: {result}")
        return result
    
    def cat_file(self, cid, output_path=None):
        """從IPFS獲取檔案
        
        Args:
            cid: 內容識別碼
            output_path: 可選的輸出檔案路徑
            
        Returns:
            檔案的二進制內容
        """
        print(f"正在從IPFS獲取CID: {cid}")
        response = requests.post(f"{self.api_url}/cat?arg={cid}")
        
        if response.status_code != 200:
            raise Exception(f"IPFS cat請求失敗: {response.text}")
            
        content = response.content
        
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(content)
                print(f"已保存內容到: {output_path}")
                
        return content
    
    def pin_add(self, cid):
        """釘住一個CID
        
        Args:
            cid: 要釘住的內容識別碼
            
        Returns:
            API響應
        """
        print(f"正在釘住CID: {cid}")
        response = requests.post(f"{self.api_url}/pin/add?arg={cid}")
        
        if response.status_code != 200:
            raise Exception(f"IPFS pin add請求失敗: {response.text}")
            
        result = response.json()
        print(f"CID已釘住: {result}")
        return result
    
    def pin_ls(self, cid=None):
        """列出釘住的內容
        
        Args:
            cid: 可選的CID過濾
            
        Returns:
            釘住內容的列表
        """
        url = f"{self.api_url}/pin/ls"
        if cid:
            url += f"?arg={cid}"
            
        response = requests.post(url)
        
        if response.status_code != 200:
            raise Exception(f"IPFS pin ls請求失敗: {response.text}")
            
        return response.json()
    
    def pin_rm(self, cid):
        """移除釘住的CID
        
        Args:
            cid: 要移除釘住的內容識別碼
            
        Returns:
            API響應
        """
        print(f"正在移除釘住的CID: {cid}")
        response = requests.post(f"{self.api_url}/pin/rm?arg={cid}")
        
        if response.status_code != 200:
            raise Exception(f"IPFS pin rm請求失敗: {response.text}")
            
        result = response.json()
        print(f"釘住已移除: {result}")
        return result
    
    def get_node_info(self):
        """獲取IPFS節點信息"""
        response = requests.post(f"{self.api_url}/id")
        
        if response.status_code != 200:
            raise Exception(f"獲取節點信息失敗: {response.text}")
            
        return response.json()
    
    def get_peers(self):
        """獲取連接的對等節點"""
        response = requests.post(f"{self.api_url}/swarm/peers")
        
        if response.status_code != 200:
            raise Exception(f"獲取對等節點失敗: {response.text}")
            
        return response.json()

def create_test_file(content, file_path):
    """創建測試檔案
    
    Args:
        content: 檔案內容
        file_path: 檔案路徑
    """
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"創建測試檔案: {file_path}")

def run_demo(ipfs, size_mb=1):
    """執行IPFS完整演示
    
    Args:
        ipfs: IPFS連接器實例
        size_mb: 測試檔案大小（MB）
    """
    print("\n=== 執行IPFS簡單MVP演示 ===")
    
    # 1. 創建測試檔案
    test_file = "test_file.txt"
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    test_content = f"這是一個測試檔案，創建於 {timestamp}\n"
    test_content += "此檔案將上傳到IPFS並下載回來，演示基本功能。\n"
    test_content += "這個簡單的MVP展示了如何與IPFS節點通信而不需要特定的客戶端庫。\n"
    
    # 創建指定大小的檔案
    if size_mb > 0:
        # 創建隨機數據來填充檔案至指定大小
        test_content += "X" * (size_mb * 1024 * 1024 - len(test_content))
    
    create_test_file(test_content[:200] + "...", test_file)  # 只顯示部分內容
    
    # 獲取起始時間
    start_time = time.time()
    
    # 2. 上傳檔案到IPFS
    print("\n--- 上傳檔案到IPFS ---")
    upload_result = ipfs.add_file(test_file)
    file_cid = upload_result['Hash']
    
    # 計算上傳時間
    upload_time = time.time() - start_time
    print(f"上傳 {size_mb}MB 檔案耗時: {upload_time:.2f} 秒")
    
    # 3. 釘住檔案
    print("\n--- 釘住檔案 ---")
    ipfs.pin_add(file_cid)
    
    # 4. 檢查釘住的檔案
    print("\n--- 檢查釘住的檔案 ---")
    pins = ipfs.pin_ls(file_cid)
    print(f"釘住內容: {json.dumps(pins, indent=2)}")
    
    # 5. 從IPFS下載檔案
    print("\n--- 從IPFS下載檔案 ---")
    download_start = time.time()
    download_file = f"downloaded_{file_cid[:6]}.txt"
    ipfs.cat_file(file_cid, download_file)
    
    # 計算下載時間
    download_time = time.time() - download_start
    print(f"下載 {size_mb}MB 檔案耗時: {download_time:.2f} 秒")
    
    # 6. 驗證下載的檔案
    print("\n--- 驗證下載的檔案 ---")
    original_size = os.path.getsize(test_file)
    downloaded_size = os.path.getsize(download_file)
    
    print(f"原始檔案大小: {original_size} 字節")
    print(f"下載檔案大小: {downloaded_size} 字節")
    
    if original_size == downloaded_size:
        print("✓ 檔案大小匹配!")
    else:
        print("✗ 檔案大小不匹配!")
    
    # 檢查文件內容
    with open(test_file, 'rb') as f1:
        content1 = f1.read()
    with open(download_file, 'rb') as f2:
        content2 = f2.read()
    
    if content1 == content2:
        print("✓ 檔案內容完全匹配!")
    else:
        print("✗ 檔案內容不匹配!")
    
    # 7. 查看IPFS節點狀態
    print("\n--- IPFS節點狀態 ---")
    try:
        peers = ipfs.get_peers()
        peer_count = len(peers.get('Peers', []))
        print(f"已連接 {peer_count} 個對等節點")
    except Exception as e:
        print(f"獲取對等節點失敗: {str(e)}")
    
    # 8. 移除釘住（可選）
    # print("\n--- 移除釘住 ---")
    # ipfs.pin_rm(file_cid)
    
    print("\n演示完成!")
    print(f"IPFS CID: {file_cid}")
    print(f"您可以訪問 http://localhost:8081/ipfs/{file_cid} 查看此文件")
    
    # 返回CID以便後續使用
    return file_cid

def main():
    parser = argparse.ArgumentParser(description='IPFS文件傳輸簡單MVP')
    parser.add_argument('--api', default='http://localhost:5001/api/v0', help='IPFS API URL')
    parser.add_argument('--action', choices=['upload', 'download', 'pin', 'unpin', 'list', 'demo'], default='demo', help='要執行的操作')
    parser.add_argument('--file', help='要上傳或下載的檔案路徑')
    parser.add_argument('--cid', help='要處理的CID')
    parser.add_argument('--output', help='下載時的輸出檔案路徑')
    parser.add_argument('--size', type=float, default=1, help='測試檔案大小(MB)')
    
    args = parser.parse_args()
    
    # 初始化IPFS連接器
    ipfs = SimpleIPFSConnector(api_url=args.api)
    
    if args.action == 'upload' and args.file:
        # 上傳檔案
        result = ipfs.add_file(args.file)
        print(f"上傳結果: {json.dumps(result, indent=2)}")
        # 釘住檔案
        ipfs.pin_add(result['Hash'])
        
    elif args.action == 'download' and args.cid:
        # 下載檔案
        output_path = args.output or f"downloaded_{args.cid[:6]}.bin"
        ipfs.cat_file(args.cid, output_path)
        
    elif args.action == 'pin' and args.cid:
        # 釘住CID
        ipfs.pin_add(args.cid)
        
    elif args.action == 'unpin' and args.cid:
        # 移除釘住
        ipfs.pin_rm(args.cid)
        
    elif args.action == 'list':
        # 列出釘住的內容
        result = ipfs.pin_ls(args.cid)
        print("釘住的內容:")
        for cid, info in result.get("Keys", {}).items():
            print(f"- {cid}: {info['Type']}")
        
    elif args.action == 'demo':
        # 執行完整演示
        run_demo(ipfs, args.size)
    
    else:
        print("請提供有效的操作和參數。使用 --help 查看幫助。")

if __name__ == "__main__":
    main()