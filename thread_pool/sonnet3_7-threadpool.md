# 改良版 `sonnet3_7-threadpool.cpp`実装のドキュメント

このドキュメントは、改良版 C++ スレッドプールの実装について、主要なコンポーネントとその機能ごとに整理した包括的な概要を提供します。

## クラス

### ThreadPool
ワーカースレッドのプールとタスクのキューを管理するメインクラスです。タスクの登録、実行、リソースの適切なクリーンアップを処理します。

## クラスのメンバ変数

### ThreadPool クラス
- `std::vector<std::thread> workers`：ワーカースレッドのコンテナ
- `std::queue<std::function<void()>> tasks`：保留中のタスクを格納するキュー
- `std::mutex queueMutex`：タスクキューへのスレッドセーフなアクセスのためのミューテックス
- `std::condition_variable condition`：スレッドの同期に使用
- `bool stop_flag`：プールがシャットダウン中であることを示すフラグ
- `std::atomic<size_t> running_threads`：アクティブなスレッド数のカウンター

### Task クラス（使用例）
- `int id`：タスク識別子
- `int duration`：ミリ秒単位の作業時間をシミュレート

## クラスのメンバ関数

### ThreadPool クラス

#### パブリックメソッド
- **コンストラクタ**：`ThreadPool(size_t numThreads = std::thread::hardware_concurrency())`
  - 指定されたスレッド数でスレッドプールを作成
  - デフォルトのスレッド数はハードウェアの同時実行数
  - `start(numThreads)` を呼び出してワーカースレッドを初期化

- **デストラクタ**：`~ThreadPool()`
  - `stop()` を呼び出し、リソースが適切にクリーンアップされるようにする
  - すべてのワーカースレッドを結合（`join`）
  - タスクキューをクリア

- **submit**：  
  ```cpp
  template<class F, class... Args> 
  auto submit(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type>
  ```
  - 関数とその引数を受け取り、タスクを作成してキューに追加
  - 関数の結果を取得するための `std::future` を返す
  - プールが停止中の場合は例外を設定
  - 例外処理を含み、ワーカースレッドのクラッシュを防ぐ

- **size**：`size_t size() const`
  - プール内のワーカースレッドの数を返す

- **pending_tasks**：`size_t pending_tasks() const`
  - キュー内の保留中のタスク数を返す

#### プライベートメソッド
- **start**：`void start(size_t numThreads)`
  - 指定された数のワーカースレッドを初期化
  - タスクを待機・実行するスレッドループを設定
  - プール停止時に適切なスレッド終了処理を行う

- **stop**：`void stop()`
  - `stop_flag` を `true` に設定
  - 待機中のスレッドをすべて通知し、フラグをチェックさせる
  - すべてのワーカースレッドを結合（`join`）
  - メモリリークを防ぐため、残っているタスクをクリア

### Task クラス（使用例）

- **コンストラクタ**：`Task(int id, int duration)`
  - タスク識別子と作業時間を初期化

- **execute**：`int execute()`
  - 指定された時間だけスリープすることで作業をシミュレート
  - タスク ID と作業時間から計算した値を返す

## 関数

### メインのサンプル関数

- **test_memory_leaks**：`void test_memory_leaks()`
  - メモリ管理のテスト（スレッドプールを作成・破棄し、タスクを保留状態にする）
  - 複数のプールを順番に作成
  - タスクを登録し、完了を待たずに破棄
  - デストラクタのリソース解放を確認

- **main**：`int main()`
  - 4スレッドのスレッドプールを作成
  - 各種タスクを登録し、機能をデモ
  - 例外処理を実演
  - 停止後のプールにタスクを登録するテスト
  - メモリリークテストの実行
  - 適切な使用パターンを示す

## テンプレート関数

- **submit テンプレート関数**：
  ```cpp
  template<class F, class... Args>
  auto submit(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type>
  ```
  - 可変テンプレートを用いて、任意の関数と引数を受け取る
  - `std::forward` による完全転送で値カテゴリを維持
  - `std::invoke_result` を使用して戻り値の型を決定
  - `std::future` を作成し、結果を取得可能にする

## 主要な設計要素

### スレッドセーフ性
- ミューテックスを使用してタスクキューへのスレッドセーフなアクセスを保証
- 条件変数を使用したスレッド同期
- アトミックカウンターによる実行中スレッドの管理

### 例外安全性
- ワーカースレッド内の例外処理でクラッシュを防ぐ
- タスク内の例外は `std::future` を通じて伝播
- 停止中のプールへの登録時の例外を適切に処理

### リソース管理
- デストラクタ内でリソースを適切に解放
- タスクキューをクリアしてメモリリークを防止
- すべてのスレッドを `join` してスレッドリークを防ぐ

### パフォーマンス考慮
- ムーブセマンティクスを使用し、不必要なコピーを回避
- 効率的なスレッド通知
- タスク登録・実行時のロックを最小限に

## 使用パターン

### 基本的な使用例
```cpp
// 4スレッドのスレッドプールを作成
ThreadPool pool(4);

// タスクを登録し、結果を取得する
auto future = pool.submit([](int value) { return value * 2; }, 42);

// 結果を取得
int result = future.get();
```

### 例外処理
```cpp
try {
    auto future = pool.submit([]() -> int {
        throw std::runtime_error("タスク失敗");
        return 42;
    });

    // 例外を再スロー
    int result = future.get();
}
catch (const std::exception& e) {
    std::cout << "例外をキャッチ: " << e.what() << std::endl;
}
```

### オブジェクト指向のタスク
```cpp
// 共有所有のタスクオブジェクトを作成
auto task = std::make_shared<Task>(1, 500);

// タスクをプールに登録
auto future = pool.submit([task]() {
    return task->execute();
});
```

## メモリ管理の改善点

- **タスクキューのクリーンアップ**：`stop` メソッド内でタスクキューを明示的にクリア
- **共有所有権**：`shared_ptr` を使用し、タスクオブジェクトの適切なメモリ管理を確保
- **値キャプチャ**：ラムダ式で参照キャプチャを避け、ダングリング参照を防止
- **例外処理**：リソースリークを防ぐための適切な例外処理
- **安全な拒否**：停止中のプールへのタスク登録時に `std::future` を例外状態に設定
