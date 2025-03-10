# シングルキュー・マルチワーカーのスレッドプール実装

このドキュメントでは、C++ の `worker_pool` テンプレートクラスについて説明します。このクラスは、固定数のワーカースレッドが単一のタスクキューを共有する形で管理するものです。以下では、各関数の説明と使用例を示します。

---

## `worker_pool` の概要

`worker_pool` はスレッドセーフな実装であり、ワーカースレッドのプールにタスクを送信できるようにします。すべてのワーカーは単一のキューを共有し、タスクを取得する方法によってラウンドロビン方式または類似の方式で分配されます。

### 主な特徴:
- **固定数のスレッド**: スレッドプールのサイズはテンプレートパラメータ `Count` によって決定される。
- **単一のキュー**: すべてのワーカーが共通のタスクキューを共有する。
- **適切な終了処理**: ワーカーは現在のタスクを完了した後、適切に終了できる。

---

## `worker_pool` クラスの関数

### 1. コンストラクタ
```cpp
worker_pool();
```
- プール内のワーカーを初期化し、それぞれのスレッドを開始する。

### 2. デストラクタ
```cpp
~worker_pool();
```
- すべてのワーカーがアイドル状態になるまで待機し、その後、適切に終了処理を行う。

### 3. `run(F&& task)`
```cpp
template<typename F>
void run(F&& task);
```
- タスクをワーカープールのキューに追加する。
- タスクはキューの末尾に追加され、すべてのワーカーが新しいタスクをチェックできるよう通知される。
- すでに終了要求が出されている場合、タスクは追加されない。

### 4. `wait_until_idle()`
```cpp
void wait_until_idle();
```
- キュー内のすべてのタスクが処理されるまで待機する。
- すべてのワーカーがアイドル状態になっていることを確認する。

### 5. `request_termination()`
```cpp
void request_termination();
```
- すべてのワーカースレッドに終了を要求する。
- 終了フラグを設定し、すべてのワーカーに通知する。

### 6. `pull()`
```cpp
std::function<void()> pull();
```
- キューの先頭のタスクを取得し、削除する。
- キューが空であるか、終了が要求されている場合は、空の関数を返す。

---

## `inner_worker` クラスの関数

`inner_worker` クラスは、プール内の個々のワーカースレッドを表します。以下にその主要な関数を説明します。

### 1. コンストラクタ
```cpp
inner_worker();
```
- 新しいスレッドを初期化し、`proc_worker()` ループ内で実行を開始する。

### 2. デストラクタ
```cpp
~inner_worker();
```
- ワーカースレッドが適切に終了するよう、アイドル状態になるまで待機し、終了要求を行い、スレッドを `join()` する。

### 3. `initialize(worker_pool* parent, int index)`
```cpp
void initialize(worker_pool* parent, int index);
```
- 親の `worker_pool` の参照とワーカーのインデックスを設定する。
- コンストラクタ後の初期化処理に使用される。

### 4. `wakeup()`
```cpp
void wakeup();
```
- ワーカーの条件変数に通知を送り、タスク処理のためにスレッドを起こす。

### 5. `wait_until_idle()`
```cpp
void wait_until_idle();
```
- 現在のタスクが完了し、ワーカーがアイドル状態になるまで待機する。

### 6. `request_termination()`
```cpp
void request_termination();
```
- 終了フラグを設定し、条件変数を通知することで、このワーカーの終了を要求する。

### 7. `proc_worker()`
```cpp
void proc_worker();
```
- ワーカースレッドのメイン処理ループ。
- キューからタスクを取得して実行し、終了要求を処理する。

---

## 使用例

以下に `worker_pool` クラスの使用例を示します。

```cpp
#include <iostream>
#include <thread>
#include "single_queue_multi_worker.cpp"

int main() {
    // 4スレッドのワーカープールを作成
    worker_pool<4> pool;

    // タスクを送信
    for (int i = 0; i < 10; ++i) {
        pool.run([i]() {
            std::cout << "タスク " << i << " がスレッド " << std::this_thread::get_id() << " によって実行されました。" << std::endl;
            // 処理のシミュレーション
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        });
    }

    // すべてのタスクが完了するまで待機
    pool.wait_until_idle();

    return 0;
}
```

---

## 注意点

- **スレッドセーフ性**: `worker_pool` および `inner_worker` のすべての公開関数は、ミューテックスと条件変数を使用することでスレッドセーフになっている。
- **終了処理**: `request_termination()` が呼び出されると、ワーカーは現在のタスクを完了した後、適切に終了する。
- **効率性の考慮**: 単一キューの設計により、タスクは送信された順に処理されるが、キューがボトルネックとなる可能性がある。

この実装は、C++ アプリケーションでスレッドプールを管理するための堅牢な基盤を提供し、予測可能な動作と適切な終了処理を実現します。