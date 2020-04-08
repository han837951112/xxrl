# xxrl

## **线段树**

```cpp
// 区间第k大
#include<bits/stdc++.h>

using namespace std;

#define ll long long
const int N = 1e5+10;
int n,m,sz; 
int root[N],a[N],b[N];
struct node{
	int l,r,sum;
}tre[N*40];

void update(int l,int r,int pre,int &now,int v){
	tre[now = ++sz].sum = tre[pre].sum+1;
	if(l==r)	return ;
	tre[now].l = tre[pre].l;	tre[now].r = tre[pre].r;
	int m = (l+r)>>1;
	if(v<=m)	update(l,m,tre[pre].l,tre[now].l,v);
	else update(m+1,r,tre[pre].r,tre[now].r,v); 
}
int query(int L,int R,int l,int r,int k){
	if(l==r)	return l;
	int m = (l+r)>>1;
	int cnt = tre[tre[R].l].sum - tre[tre[L].l].sum;
	if(k <= cnt)	return query(tre[L].l,tre[R].l,l,m,k);
	else return query(tre[L].r,tre[R].r,m+1,r,k-cnt);
}

int main(){
	int t = 1,l,r,k,n,q;
	while(t--){
		scanf("%d%d",&n,&q);
		for(int i=1;i<=n;++i)	scanf("%d",&a[i]),b[i]=a[i];
		sort(b+1,b+1+n);
		int end = unique(b+1,b+1+n) - b - 1;
		for(int i=1;i<=n;++i)	a[i] = lower_bound(b+1,b+1+end,a[i])-b, update(1,end,root[i-1],root[i],a[i]); 
    	for(int i=1;i<=q;++i){
			scanf("%d%d%d",&l,&r,&k);
			printf("%d\n",b[query(root[l-1],root[r],1,end,k)]);
		}
	}
}

// 两颗线段树求区间覆盖
#include<bits/stdc++.h>
using namespace std;
#define ls(p) (p<<1)
#define rs(p) (p<<1|1)
#define ll long long
const int N = 1e5+10;

int n,m;
struct node{
    int l,r,lm,rm,sm;
}a[N<<2],b[N<<2];

void push_up(int rt,int l,int r){
    int m = (l+r) >> 1;
    a[rt].lm = a[ls(rt)].lm;
    a[rt].rm = a[rs(rt)].rm;
    a[rt].sm = max(max(a[ls(rt)].sm,a[rs(rt)].sm),a[rs(rt)].lm+a[ls(rt)].rm);
    if(a[ls(rt)].lm == m-l+1)   a[rt].lm += a[rs(rt)].lm;
    if(a[rs(rt)].rm == r-m  )   a[rt].rm += a[ls(rt)].rm;
    b[rt].lm = b[ls(rt)].lm;
    b[rt].rm = b[rs(rt)].rm;
    b[rt].sm = max(max(b[ls(rt)].sm,b[rs(rt)].sm),b[rs(rt)].lm+b[ls(rt)].rm);
    if(b[ls(rt)].lm == m-l+1)   b[rt].lm += b[rs(rt)].lm;
    if(b[rs(rt)].rm == r-m  )   b[rt].rm += b[ls(rt)].rm; 
}
void push_down(int rt,int l,int r){
    int m = (l+r)>>1;
    if(a[rt].sm == r-l+1){
        a[ls(rt)].lm = a[ls(rt)].rm = a[ls(rt)].sm = m-l+1;
        a[rs(rt)].lm = a[rs(rt)].rm = a[rs(rt)].sm = r - m; 
    }else if(a[rt].sm == 0){
        a[ls(rt)].lm = a[ls(rt)].rm = a[ls(rt)].sm = 0;
        a[rs(rt)].lm = a[rs(rt)].rm = a[rs(rt)].sm = 0;
    }
    if(b[rt].sm == r-l+1){
        b[ls(rt)].lm = b[ls(rt)].rm = b[ls(rt)].sm = m-l+1;
        b[rs(rt)].lm = b[rs(rt)].rm = b[rs(rt)].sm = r - m; 
    }else if(b[rt].sm == 0){
        b[ls(rt)].lm = b[ls(rt)].rm = b[ls(rt)].sm = 0;
        b[rs(rt)].lm = b[rs(rt)].rm = b[rs(rt)].sm = 0;
    }
}

void build(int rt,int l,int r){
    a[rt].lm = b[rt].lm = a[rt].rm = b[rt].rm = 
        a[rt].sm = b[rt].sm = r-l+1;
    a[rt].l = b[rt].l = l;
    a[rt].r = b[rt].r = r;
    if(l == r)  return ;
    int m = (l+r) >> 1;
    build(ls(rt),l,m);  build(rs(rt),m+1,r);
}

void update(int op,int l,int r,int rt){
    if(a[rt].l >= l && a[rt].r <= r){
        if(op == 1){
            a[rt].lm = a[rt].rm = a[rt].sm = 0;
        }else if(op == 0){
            a[rt].lm = a[rt].rm = a[rt].sm = 0;
            b[rt].lm = b[rt].rm = b[rt].sm = 0;
        }else{
            a[rt].lm = a[rt].rm = a[rt].sm = a[rt].r-a[rt].l+1;
            b[rt].lm = b[rt].rm = b[rt].sm = b[rt].r-b[rt].l+1;    
        }
        return ;
    }
    int m = (a[rt].l+a[rt].r)>>1;
    push_down(rt,a[rt].l,a[rt].r);
    if(l <= m)  update(op,l,r,ls(rt));
    if(r > m )  update(op,l,r,rs(rt));
    push_up(rt,a[rt].l,a[rt].r);
}
int query(int op,int rt,int len){
    if(a[rt].l == a[rt].r)  return a[rt].l;
    push_down(rt,a[rt].l,a[rt].r);
    int m = (a[rt].l+a[rt].r) >> 1;
    if(op){
        if(a[ls(rt)].sm >= len) return query(op,ls(rt),len);
        else if(a[ls(rt)].rm + a[rs(rt)].lm >= len)
            return m - a[ls(rt)].rm + 1;
        else return query(op,rs(rt),len);
    }else{
        if(b[ls(rt)].sm >= len) return query(op,ls(rt),len);
        else if(b[ls(rt)].rm + b[rs(rt)].lm >= len)
            return m - b[ls(rt)].rm + 1;
        else return query(op,rs(rt),len);
    }
}
char op[20];
void solve(int cas){
    printf("Case %d:\n",cas);
    scanf("%d%d",&n,&m);
    build(1,1,n);
    int l,r;
    for(int i=1;i<=m;++i){
        scanf("%s%d",op,&l);
        if(op[0]=='D'){
            if(a[1].sm >= l){
                int pos = query(1,1,l);
                printf("%d,let's fly\n",pos);
                update(1,pos,pos+l-1,1);
            }else{
                puts("fly with yourself");
            }
        }else if(op[0]=='N'){
            if(a[1].sm>=l){
                int pos = query(1,1,l);
                printf("%d,don't put my gezi\n",pos);
                update(0,pos,pos+l-1,1);
            }else if(b[1].sm>=l){
                int pos = query(0,1,l);
                printf("%d,don't put my gezi\n",pos);
                update(0,pos,pos+l-1,1);
            }else{
                puts("wait for me");
            }
        }else{
            scanf("%d",&r);
            update(2,l,r,1);
            printf("I am the hope of chinese chengxuyuan!!\n");
        }
    }
}
int main(){
    int t;  scanf("%d",&t);
    for(int i=1;i<=t;++i)   solve(i);
    return 0;
}


// 区间加,乘,求和
#include<bits/stdc++.h>
#define ll long long
#define pii pair<int,int>
#define ls(p) (p<<1)
#define rs(p) (p<<1|1)
using namespace std;
const int N = 1e5+10;
ll a[N],modp;
struct node{
    int  l,r;
    ll val,tag,mut;
    int len(){return r-l+1;}
}sgt[N<<2];

void push_up(int rt){
    sgt[rt].val = (sgt[ls(rt)].val + sgt[rs(rt)].val)%modp;
}
void build(int rt,int l,int r){
    sgt[rt].l = l ; sgt[rt].r = r;  sgt[rt].mut = 1;
    if(l==r){sgt[rt].val = a[l];    return ;}
    int m = (l+r)>>1;
    build(ls(rt),l,m);  build(rs(rt),m+1,r);
    push_up(rt);
}

void push_down(int rt){
    if(sgt[rt].mut != 1){
        sgt[ls(rt)].val = (sgt[rt].mut * sgt[ls(rt)].val)%modp;
        sgt[rs(rt)].val = (sgt[rt].mut * sgt[rs(rt)].val)%modp;
        sgt[ls(rt)].mut = (sgt[rt].mut * sgt[ls(rt)].mut)%modp;
        sgt[rs(rt)].mut = (sgt[rt].mut * sgt[rs(rt)].mut)%modp;
        sgt[ls(rt)].tag = (sgt[rt].mut * sgt[ls(rt)].tag)%modp;
        sgt[rs(rt)].tag = (sgt[rt].mut * sgt[rs(rt)].tag)%modp;
        sgt[rt].mut = 1;
    }
    if(sgt[rt].tag){
        sgt[ls(rt)].val = (sgt[rt].val* sgt[ls(rt)].len() + sgt[ls(rt)].val)%modp;
        sgt[rs(rt)].val = (sgt[rt].val* sgt[rs(rt)].len() + sgt[rs(rt)].val)%modp;
        sgt[ls(rt)].tag = (sgt[rt].tag + sgt[ls(rt)].tag) %modp;
        sgt[rs(rt)].tag = (sgt[rt].tag + sgt[rs(rt)].tag) %modp;
        sgt[rt].tag = 0;
    }
}

void update(int rt,int l,int r,ll val){
    if(sgt[rt].l >= l && sgt[rt].r <=r ){
        sgt[rt].tag = (val + sgt[rt].tag)%modp;
        sgt[rt].val = (val * sgt[rt].len() + sgt[rt].val) %modp;
        return ; 
    }
    if(sgt[rt].mut !=1 || sgt[rt].tag)  push_down(rt);
    int m = (sgt[rt].l + sgt[rt].r) >> 1;
    if(l<=m)  update(ls(rt),l,r,val);
    if(r>m) update(rs(rt),l,r,val);
    push_up(rt);
}
void updateMut(int rt,int l,int r,ll mut){
    if(sgt[rt].l >= l && sgt[rt].r <= r){
        sgt[rt].mut = (sgt[rt].mut * mut)%modp;
        sgt[rt].tag = (sgt[rt].tag * mut)%modp;
        sgt[rt].val = (sgt[rt].val * mut)%modp;
        return ;
    }
    if(sgt[rt].mut != 1|| sgt[rt].tag)  push_down(rt);
    int m = (sgt[rt].l + sgt[rt].r )>>1;
    if(l<=m)  updateMut(ls(rt),l,r,mut);
    if(r>m)   updateMut(rs(rt),l,r,mut);
    push_up(rt);
}

ll query(int rt,int l,int r){
    if(sgt[rt].l >= l && sgt[rt].r <=r )
        return sgt[rt].val % modp;
    if(sgt[rt].mut !=1 || sgt[rt].tag)  push_down(rt);
    int m = (sgt[rt].l + sgt[rt].r )>>1;
    ll res = 0;
    if(l<=m)  res += query(ls(rt),l,r);
    if(r<m)   res += query(rs(rt),l,r);
    return res %modp;
}
ll n,m;
int main(){
    scanf("%lld%lld%lld",&n,&m,&modp);
    for(int i=1;i<=n;++i)   scanf("%lld",&a[i]);
    build(1,1,n);
    ll op,x,y,k;
    for(int i=1;i<=m;++i){
        scanf("%lld%lld%lld",&op,&x,&y);
        if(op==1){
            scanf("%lld",&k);
            updateMut(1,x,y,k);
        }else if(op==2){
            scanf("%lld",&k);
            update(1,x,y,k);
        }else{
            printf("%lld\n",query(1,x,y));
        }
    }
    return 0;
}
```

## **图论**

```cpp
// 前向星存图
struct Edge{
    int w,v,next;
}edge[M];
int tot,head[N];

void init(){
    tot = 0;
    memset(head,0,sizeof head);
}
void add(int u,int v,int w){
    edge[++tot].w = w;
    edge[tot].v = v;
    edge[tot].next = head[u];
    head[u] = tot;
}


// dijkstra 最短路
typedef  pair<int,int> pii;
void dijkstra(int s){
    fill(d,d+n+1,inf);
    priority_queue<pii,vector<pii>,greater<pii> > que;
    d[s] = 0;   que.push(pii(0,s));
    while(que.size()){
        pii p = que.top();  que.pop();
        int u = p.second;
        if(d[u]<p.first)    continue;   
        for(int i=head[u];i;i=edge[i].next){
            int v = edge[i].v, w = edge[i].w;
            if(d[v]>d[u]+w){
                d[v] = d[u]+w;
                que.push(pii(d[v],v));
            }
        }
    }
}

// spfa 判负环, cnt代表入队次数, 大于n则为负环
// 理论复杂度O(n^3) ?
bool spfa(){
    fill(d,d+n+1,inf);
    memset(v,0,sizeof(v));
    memset(cnt,0,sizeof(cnt));
    queue<int> q;
    d[1] = 0;   cnt[1] = 0; v[1] = 1;  q.push(1);
    while(q.size()){
        int x = q.front();  q.pop();
        v[x] = 0;
        for(int i=head[x];i!=-1;i=edge[i].next){
            int y = edge[i].to;
            int z = edge[i].w;
            if(d[y]>d[x]+z){
                d[y] = d[x]+z;
                cnt[y] = cnt[x] + 1;
                if(cnt[y]>n)
                    return true;
                if(v[y]==0){
                    q.push(y);
                    v[y] =1;
                }   
            }
        }
    }
    return false;
}

/*
 *
 * tarjan 系列
 *
 */
int dfn[N],low[N];

// 割点与桥
void tarjan(int u,int fa){
    int v;
    int child = 0,k = 0;
    dfn[u] = low[u] = ++dfs_clock;
    for(int i=head[u];i!=-1;i=edge[i].nxt){
        v = edge[i].to;
        if(v == fa && !k){  // 处理重边 第一次访问到父亲则不访问
            k++;
            continue;
        }
        if(!dfn[v]){        // v是u的儿子
            child++;
            tarjan(v,u);
            low[u] = min(low[u],low[v]); // 儿子能访问到,父亲也能访问到
            if(low[v] > dfn[u]){    // 儿子不能访问到父亲前面的点
                // is_cut[i] = true;    // i号边为割边
            }
            if(low[v] >= dfn[u] && fa!=-1){ // 非树根且儿子不能访问他前面的点
                iscutpoint[u] = true;
            }
        }else{              // v已经被访问过, 即u可以绕过父亲来访问更靠前的点
            low[u] = min(low[u],dfn[v]);
        }
    }
    if(fa == -1 && child>1 ){   // 有超过两个儿子的树根一定是割点
        iscutpoint[u] = true;
    }
}

// 边双缩点同无向图
void tarjan(int u,int fa){
    dfn[u] = low[u] = ++stemp;
    s.push(u); in[u] = true;
    int son = 0,k = 1;
    for(int i=head[u];i;i=e[i].nxt){
        int v = e[i].to;
        if(v==fa && k){k = 0; continue;}    //   处理重边
        if(!dfn[v]){
            tarjan(v,u);
            low[u] = min(low[u],low[v]);
        }else if(in[v]){
            low[u] = min(low[u],dfn[v]);
        }
    }
    if(low[u] == dfn[u]){
        int v; ++cnt_block;
        do{
            v = s.top();    s.pop();
            belong[v] = cnt_block;
        }while(v!=u);
    }
}

// 点双
void tarjan(int u,int fa){
    dfn[u] = low[u] = ++stemp;
    s.push(u);
    int son = 0,k = 1;
    for(int i=head[u];i;i=e[i].nxt){
        int v = e[i].to;
        if(v==fa && k){k = 0; continue;}    //   处理重边
        if(!dfn[v]){
            tarjan(v,u);
            low[u] = min(low[u],low[v]);
            if(low[v] >= dfn[u]){
                if(fa != -1)    cut[u] = 1;
                blocks[++cnt_block].clear();
                blocks[cnt_block].push_back(u); // 加入u
                int now;
                do{
                    now = s.top();  s.pop();
                    blocks[cnt_block].push_back(now);
                }while(now != v);               //栈弹到v,并加入双联通分量中
            }
        }else{
            low[u] = min(low[u],dfn[v]);
        }
    }
    if(fa==-1 && son>1) cut[u] = 1;
}



/*
 * 
 * 流
 *  
 */
// hk求最大匹配
int Mx[N],My[N];
// Mx 表示左集合顶点所匹配的右集合顶点序号, My 相反
int dx[N],dy[N];
// dx 表示左集合i顶点距离编号 dy 为右集合
int used[N];
int dis;
vector<int> g[N];
 // 寻找增广路径集,每次只寻找当前最短的增广路
bool SearchP(){
    queue<int> Q;
    dis = INF;
    memset(dy,-1,sizeof dy);
    memset(dx,-1,sizeof dx);
    for(int i=1;i<=uN;++i){
        if(Mx[i] == -1){    //将未匹配的节点入队,并初始化其距离为0
            Q.push(i);
            dx[i] = 0;
        }
    }
    while(!Q.empty()){
        int u = Q.front();
        Q.pop();
        if(dx[u] > dis) break;
        int sz = g[u].size();
        for(int i=0;i<sz ;++i){
            int v = g[u][i];
            if(dy[v] == -1){
                dy[v] = dx[u] +1;
                if(My[v] == -1) dis = dy[v];    // 找到了一条增广路,dis为增广路终点的编号
                else{
                    dx[My[v]] = dy[v]+1;
                    Q.push(My[v]);
                }
            }
        }
    }
    return dis!=INF;
}
bool DFS(int u){
    int sz = g[u].size();
    for(int i=0;i<sz;++i){
        int v = g[u][i];
        if(!used[v] && dy[v] == dx[u]+1){   // 当前节点没有匹配,且距离上一节点+1
            used[v] = true;
            if(My[v] != -1 && dy[v]==dis)   continue;   // v已经被匹配且已到所有存在的增广路终点的编号,不可能存在增广路
            if(My[v] == -1 || DFS(My[v])){
                My[v] = u;
                Mx[u] = v;
                return true;
            }
        }
    }
    return false;
}
int MaxMatch(){
    int res = 0;
    memset(Mx,-1,sizeof Mx);
    memset(My,-1,sizeof My);
    while(SearchP()){
        memset(used,0,sizeof used);
        for(int i=1;i<=uN;++i){
            if(Mx[i]==-1 && DFS(i)){
                res++;
            }
        }
    }
    return res;
}

// dinic最大流
struct E{
    int u,v,flow,nxt;
    E(){}
    E(int u,int v,int flow,int nxt):u(u),v(v),flow(flow),nxt(nxt){}
}e[N*2];

int n,m,sp,tp,tot;
int head[N],dis[N];
void init(){
    tot = 0;    memset(head,-1,sizeof head);
}
void addE(int u,int v,int flow){
    e[tot].u = u; e[tot].v = v; e[tot].flow = flow; e[tot].nxt = head[u]; head[u] = tot++;
    e[tot].u = v; e[tot].v = u; e[tot].flow = 0; e[tot].nxt = head[v]; head[v] = tot++;
}
int q[N];
int bfs(){
    int qtop = 0,qend=0;
    memset(dis,-1,sizeof dis);  
    dis[sp] = 0;    
    q[qend++] = sp;
    while(qtop!=qend){
        int u = q[qtop++];
        if(u==tp)   return true;
        for(int i=head[u];~i;i=e[i].nxt){
            int v = e[i].v;
            if(dis[v]==-1 && e[i].flow){
                dis[v] = dis[u]+1;  
                q[qend++] = v;
            }
        }
    }
    return dis[tp]!=-1;
}
int dfs(int u,int flow){
    int res = 0;
    if(u==tp)   return flow;
    for(int i=head[u];i!=-1&&flow;i=e[i].nxt){
        int v = e[i].v;
        if(dis[v]==dis[u]+1 && e[i].flow){
            int d = dfs(v,min(e[i].flow,flow));
            e[i].flow -=d;
            e[i^1].flow += d;
            res+=d;
            flow -= d;
        }
    }
    if(!res)
        dis[u] = -2;
    return res;
}
int dinic(){
    int ans=0;
    while(bfs()){
        ans+=dfs(sp,INF);
    }
    return ans;
}

```

**VI大数**

```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;
typedef pair<int,int> pii;
typedef vector<int> VI;


VI maxV(VI A,VI B){
    if(A.size()!=B.size()){
        return A.size()>B.size()?A:B;
    }
    for(int i=A.size()-1;i>=0;--i){
        if(A[i]!=B[i])  return A[i]>B[i]?A:B ;
    }
    return A;
}
bool cmp(VI A,VI B){
    if(A.size()!=B.size())  return A.size() < B.size();
    reverse(A.begin(),A.end());
    reverse(B.begin(),B.end());
    return A<B;
}

namespace bigI{
    const int bas = 1e4;    // 压位优化 1位代表四位十进制
    void print(VI A){
        if(A.size()==0) return ;
        printf("%d",A.back());  // uncheck A is empty
        for(int i=A.size()-2;i>=0;--i)  printf("%04d",A[i]); // 长度为每一位代表的长度
        puts("");   // 换行
    }
// 整数加法就先用大数1成 转成大数加
    VI add(VI A,VI B){
        static VI C;
        C.clear();
        for(int i=0,t=0;i<(int)A.size()||i<(int)B.size() || t;++i){
            if(i<(int)A.size())  t+=A[i];
            if(i<(int)B.size())  t+=B[i];
            C.push_back(t%bas); t/=bas;
        }
        return C;
    }
    // 手写不保证正确
    VI sub(VI A,VI B){
        int sign = 0;
        if(cmp(A,B))    swap(A,B),sign = 1;
        for(int i=0,t=0;i<(int)B.size()||t;++i){
            if(i<(int)B.size()) t= t-B[i];
            if(i<(int)A.size()) t+= A[i];
            if(t<0){
                A[i] = t+bas;
                t = -1;
            }else{
                A[i] = t;
                t = 0;
            }
        }
        while(A.size() && A.back()==0)  A.pop_back();
        if(sign && A.size())    A.back()  *= -1; 
        return A;
    }
    VI mul(VI A,ll b){
        static VI C;
        C.clear();
        ll t = 0;
        for(int i=0;i<(int)A.size()||t;++i){
            if(i<(int)A.size())  t+=A[i]*b;
            C.push_back(t%bas); t/=bas;
        }
        return C;
    }
    // 自己写的 可能有错
    VI div(VI A,ll B){
        static VI C;
        C.clear();
        for(int i = (int)A.size()-1,r = 0;i>=0;--i){
            r = r*bas + A[i];
            C.push_back(r/B);   r%=B;
        }
        reverse(C.begin(),C.end());
        while((int)C.size()>1 && !C.back())  C.pop_back();
        return C;
    }
    // a%b = a - a/b*b;
    VI mod(VI A,ll B){
        return sub(A,mul(div(A,B),B));
    }
}

namespace big{
    const int bas = 1e1;    // 10进制
    void print(VI A){
        for(int i=(int)A.size()-1;i>=0;--i) printf("%d",A[i]);
        puts("");   // 换行
    }
    // 四则运算同上
}

using namespace bigI;

int main(){
    VI res(1);  res[0] = 0;
    VI meta(1); meta[0] = 1;    // 单位元
    int n,op;  cin >> n;
    ll val;
    for(int i=1;i<=n;++i){
        cin >> op >> val;
        switch (op)
        {
            case 0:     // add
                res = add(res,mul(meta,val));
                break;
            case 1:     // sub
                res = sub(res,mul(meta,val));
                break;
            case 2:     // mul
                res = mul(res,val);
                break;
            case 3:     // div
                res = div(res,val);
                break;
            default:
                break;
        }
    }
    print(res);
    return 0;
}

```


## **树链剖分**

染色

```cpp
#include<bits/stdc++.h>
#define lowbit(x)   (x&(-x))
using namespace std;
#define lson(p) (p<<1)
#define rson(p) (p<<1|1)
#define ll long long
const int N = 1e5+10;


struct Edge{
    int to,next,w;
}edge[N*2];

int head[N],tot;
int top[N];  // 所在重链的顶端节点
int fa[N];   // 父亲
int deep[N]; // 深度
int num[N];  // 儿子个数
int p[N];    // 在dfs序的位置  
int fp[N];   // 位置节点号的反向映射
int son[N]; // 重儿子
int pos;
int Lc,Rc;
void addedge(int u,int v,int w=0){
    edge[tot].to = v;
    edge[tot].next = head[u];
    edge[tot].w = w;
    head[u] = tot++;
}

void init(){
    memset(head,-1,sizeof(head));
    memset(son,-1,sizeof(son));
    tot = 0;
    pos = 1;
}

 //第一遍dfs   处理fa,num,deep,son
void dfs1(int u,int pre,int d){
    deep[u] = d;
    fa[u] = pre;
    num[u] = 1;
    for(int i=head[u];i!=-1;i=edge[i].next){
        int v = edge[i].to;
        if(v!=pre){
            dfs1(v,u,d+1);
            num[u] += num[v];
            if(son[u] == -1 || num[v] > num[son[u]])
                son[u] = v;
        }
    }
}
// 第二遍dfs  处理 top,p,fp
void dfs2(int u,int sp){
    top[u] = sp;        
    p[u] = pos++;
    fp[p[u]] = u;
    if(son[u]== -1)    return ;
    dfs2(son[u],sp);    // 当前链继续走重儿子
    for(int i=head[u];i!=-1;i=edge[i].next){
        int v = edge[i].to;
        if( v!= son[u] && v!=fa[u])
            dfs2(v,v);  // 以自己为链首的新链
    }
}

int a[N];
struct node{
    int l,r;
    int lc,rc;
    int sum,tag;
}seg[N*4];

void push_up(int p){
    seg[p].lc = seg[lson(p)].lc;    seg[p].rc = seg[rson(p)].rc;
    seg[p].sum = seg[lson(p)].sum + seg[rson(p)].sum;
    if(seg[lson(p)].rc == seg[rson(p)].lc) seg[p].sum--;
}
void push_down(int rt){
    if(seg[rt].tag){
        seg[lson(rt)].tag = seg[rson(rt)].tag = seg[rt].tag;
        seg[lson(rt)].sum = seg[rson(rt)].sum = 1;
        seg[lson(rt)].lc = seg[lson(rt)].rc = seg[rt].lc;
        seg[rson(rt)].lc = seg[rson(rt)].rc = seg[rt].lc;
        seg[rt].tag = 0;
    }
}

void build(int pp,int l,int r){
    seg[pp].l = l;
    seg[pp].r = r;
    if(l==r){
        return;
    }
    int mid = (l+r)>>1;
    build(lson(pp),l,mid);
    build(rson(pp),mid+1,r);
}

void update(int p,int l,int r,int val){
    if(seg[p].l >= l && seg[p].r <= r){
        seg[p].sum = seg[p].tag = 1;
        seg[p].lc = seg[p].rc = val;
        return;
    }
    push_down(p);
    int mid = (seg[p].r+seg[p].l)>>1;
    if(l<=mid)  update(lson(p),l,r,val);
    if(r>mid)   update(rson(p),l,r,val);
    push_up(p);
}

int query(int p,int l,int r,int L,int R){
    if(seg[p].l == L)   Lc = seg[p].lc;
    if(seg[p].r == R)   Rc = seg[p].rc;
    if(seg[p].l >= l && seg[p].r <= r){
        return seg[p].sum;
    }
    push_down(p);
    int mid = ( seg[p].l + seg[p].r ) >> 1;
    if(r <= mid)    return query(lson(p),l,r,L,R);
    else if(l > mid )   return query(rson(p),l,r,L,R);
    else{
        int ans = query(lson(p),l,mid,L,R) + query(rson(p),mid+1,r,L,R);
        if(seg[lson(p)].rc == seg[rson(p)].lc)  ans--;
        return ans;
    }
}


int fquery(int u,int v){
    int ans = 0;
    int ans1=-1,ans2=-1;
    int tu = top[u], tv = top[v];
    while(tu != tv){
        if(deep[tu] < deep[tv]){
            swap(tu,tv);
            swap(u,v);
            swap(ans1,ans2);
        }
        ans += query(1,p[tu],p[u],p[tu],p[u]);
        if(Rc == ans1)  ans--;
        ans1 = Lc;  u = fa[tu]; tu = top[u];
    }
    if(deep[u] < deep[v])   swap(u,v),swap(ans1,ans2);
    ans += query(1,p[v],p[u],p[v],p[u]);
    if(Rc == ans1)  ans--;
    if(Lc == ans2)  ans--;
    return ans;
}
void fupdate(int u,int v,int c){
    int tu = top[u], tv = top[v];
    while(tu != tv){
        if(deep[tu] < deep[tv]){
            swap(tu,tv);
            swap(u,v);
        }
        update(1,p[tu],p[u],c);
        u = fa[tu]; tu = top[u];
    }
    if(deep[u] > deep[v])   swap(u,v);
    update(1,p[u],p[v],c);
}
int n,m;    char op[10];
int main(){
    scanf("%d%d",&n,&m);
    int fr,to,w;
    init();
    for(int i=1;i<=n;++i){
        scanf("%d",&a[i]);
    }
    for(int i=1;i<=n-1;++i){
        scanf("%d%d",&fr,&to);
        addedge(fr,to);
        addedge(to,fr);
    }
    dfs1(1,0,0);
    dfs2(1,1);
    build(1,1,n);
    for(int i=1;i<=n;++i)   update(1,p[i],p[i],a[i]);
    int u,v,c;
    while(m--){
        scanf("%s%d%d",op,&u,&v);
        if(op[0]=='C'){
            scanf("%d",&c);
            fupdate(u,v,c);
        }else{
            printf("%d\n",fquery(u,v));
        }
    }
    return 0;
}

```

****************************************************

add : 10-5

## **扫描线**

统计矩形覆盖两次以上面积

```cpp
#include<bits/stdc++.h>
#define ll long long
#define ls(i) i<<1
#define rs(i) i<<1|1
using namespace std;
const int N = 2e5+10;
struct node{
	double l,r;		// 左右下标
	double len;		// 区间长度
	int lf,rf;
	int cover;		// 被覆盖多少次
}sgt[N<<3];
struct L{
	double x,y1,y2;	// 垂直x轴的线段
	int state;		// 入边/出边
	bool operator<(L oth)const{
		if(x==oth.x)	return state > oth.state;
		return x < oth.x;
	}
}line[N];
double y[N];
void pushup(int rt){
	if(sgt[rt].cover >1){	// 被完全覆盖
		sgt[rt].len = sgt[rt].r - sgt[rt].l;
	}else if(sgt[rt].lf+1==sgt[rt].rf){
		sgt[rt].len = 0;
	}
	else{							
		sgt[rt].len = sgt[ls(rt)].len + sgt[rs(rt)].len;
	}
}
// 建树
void build(int l,int r,int rt=1){
	sgt[rt].l = y[l];	sgt[rt].r = y[r];	sgt[rt].cover = 0;
	sgt[rt].lf = l;	sgt[rt].rf = r;
	if(l+1>=r)	return ;
	int mid = (l+r)>>1;
	build(l,mid,ls(rt));
	build(mid,r,rs(rt));
}
// 加边
void modify(double yl,double yr,int op,int rt=1){
	double lf = sgt[rt].l, rf = sgt[rt].r;
	if(yl<=lf && yr>=rf && sgt[rt].lf+1 == sgt[rt].rf){	// 被覆盖
		sgt[rt].cover += op;
		pushup(rt);
		return ;
	}
	if(yl<sgt[ls(rt)].r)	modify(yl,yr,op,ls(rt));	// 落入左儿子
	if(yr>sgt[rs(rt)].l)	modify(yl,yr,op,rs(rt));	// 落入右儿子
	pushup(rt);
}

int main(){
	int t;	scanf("%d",&t);
	while(t--){
		int n;	double x1,x2,y1,y2;
		scanf("%d",&n);
		for(int i=1;i<=n;++i){
			scanf("%lf%lf%lf%lf",&x1,&y1,&x2,&y2);
			y[i] = y1;	y[i+n] = y2;
			line[i]=(L){x1,y1,y2,1};
			line[i+n]=(L){x2,y1,y2,-1};
		}
		sort(y+1,y+1+n*2);
		sort(line+1,line+1+n*2);
		int m = unique(y+1,y+1+n*2)-y-1;// 离散化 获得y值个数
		build(1,m,1);	// 根据y的个数建树
		double ans = 0;
		for(int i=1;i<=n*2;++i){	// 扫描
			ans += sgt[1].len  * (line[i].x - line[i-1].x);
			modify(line[i].y1,line[i].y2,line[i].state,1);
		}
		printf("%.2lf\n",ans);
	}
}

```

## **kruskal**

最小生成树

```cpp
struct E{
    int u,v;
    ll w;
}e[M];
bool cmp(e a,e b)
{
    return a.w<b.w;
}
int fath[N]; //我们用并查集来维护当前两个点是否联通
int n,m;
void init()
{
    for(int i=0;i<n;++i) 
        fath[i] = i;
}
int find(int f)
{
    if(f==fath[f])  return f;
    return fath[f] = find(fath[f]);
}

void merge(int a,int b)
{
    int u = find(a), v = find(b);
    fath[u] = v;
}
ll Kruskal()
{
    init();
    sort(e+1,e+1+m,cmp);
    ll ans = 0;
    for(int i=1;i<=m;++i)
    {
        if(find(e[i].u)==find(e[i].v))    continue;//如果联通,则跳过当前边
        merge(e[i].u,e[i].v);  //两个点的集合合并
        ans+=e[i].w;
    }
    return ans;
}
```

## **最小树形图**

无向图的最小生成树 复杂度大概n^3

```cpp
#include<bits/stdc++.h>
#define ll long long
#define ls(r)   r<<1
#define rs(r)   r<<1|1
using namespace std;
const int N = 1e3+10;
const int INF = 0x3f3f3f3f;

struct edge{
    int u,v,w;
    edge(int u,int v,int w):u(u),v(v),w(w){}
    edge(){}
};

struct directed_MT{
    int n,m;
    edge e[N*10];
    int vis[N],pre[N],id[N],in[N];
    void init(int _n){n = _n; m = 0;}
    void add(int u,int v,int w){e[m++] = edge(u,v,w);}

    int dirMt(int sp){
        int ans = 0;
        while(1){
            for(int i=0;i<n;++i)    in[i] = INF;
            for(int i=0;i<m;++i){
                int u = e[i].u;
                int v = e[i].v;
                // 更新v的距离，和前驱
                if(e[i].w < in[v] && u!=v){
                    in[v] = e[i].w;
                    pre[v] = u;
                }
            }
            for(int i=0;i<n;++i){
                if(i==sp) continue;
                // 无法到达 i 点 不连通
                if(in[i] == INF)    return -1;
            }
            int cnt = 0;    // 新图点的数量
            memset(id,-1,sizeof id);
            memset(vis,-1,sizeof vis);
            in[sp] = 0;
            for(int i=0;i<n;++i){
                ans += in[i];   // 加上到这个点的距离
                int v = i;
                // 找自环
                while(vis[v]!=i && id[v]==-1 && v!=sp){
                    vis[v] = i;
                    v = pre[v];
                }
                if(v!=sp && id[v] ==-1){    // 缩点
                    for(int u=pre[v];u!=v;u=pre[u]) 
                        id[u] = cnt;
                    id[v] = cnt++;
                }
            }
            if(cnt==0)  break;  // 没有自环 算法截止
            for(int i=0;i<n;++i)    // 一个点构成一个环
                if(id[i]==-1)   id[i] = cnt++;
            for(int i=0;i<m;++i){  // 重新构图
                int v = e[i].v;
                e[i].v = id[e[i].v];
                e[i].u = id[e[i].u];
                if(e[i].v != e[i].u)
                    e[i].w -= in[v];    // 已经可以到达v，所以到v的距离要减去
            }
            n = cnt;
            sp = id[sp];
        }
        return ans;
    }
}MT;


void solve(){
    int n,m,sp;
    scanf("%d%d%d",&n,&m,&sp);
    MT.init(n);
    int u,v,w;
    for(int i=1;i<=m;++i){
        scanf("%d%d%d",&u,&v,&w);
        MT.add(u,v,w);
    }
    int ans = MT.dirMt(sp);
    if(ans == -1)   puts("impossible");
    else  printf("%d\n",ans);
}
int main(){
    int t; scanf("%d",&t);
    for(int i=1;i<=t;++i){
        printf("Case %d: ",i);
        solve();
    }
    return 0;
}
```

## **fhqTreap**

用于统计节点排名,前驱后继,排名值的不带旋转的平衡树

```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;
typedef pair<int,int> pii;

mt19937 rnd(233);   // 随机种子
const int N = 1e5+5;
struct node{
    int l,r,val,key,size;
    // key为随机数，保证平衡性
}fhq[N];
int cnt,root;
// 新节点，返回其编号
inline int newnode(int val){
    fhq[++cnt].val = val;   fhq[cnt].key = rnd();   fhq[cnt].size = 1;
    return cnt;
}
// 更新size
inline void update(int rt){
    fhq[rt].size = fhq[fhq[rt].l].size + fhq[fhq[rt].r].size + 1; 
}
// 分裂
void split(int rt,int val,int &x,int &y){
    if(!rt){
        x = y = 0;  return;
    }
    if(fhq[rt].val <= val){
        x = rt;
        split(fhq[rt].r,val,fhq[rt].r,y);
    }else{
        y = rt;
        split(fhq[rt].l,val,x,fhq[rt].l);
    }
    update(rt);
}
// 合并
int merge(int x,int y){
    if(!x || !y)    return x+y;
    if(fhq[x].key <= fhq[y].key){
        fhq[x].r = merge(fhq[x].r,y);
        update(x);
        return x;
    }else{
        fhq[y].l = merge(x,fhq[y].l);
        update(y);
        return y;
    }
}
int x,y,z;
// 插入
inline void insert(int val){
    split(root,val,x,y);
    root = merge(merge(x,newnode(val)),y);
}
// 删除
inline void del(int val){
    split(root,val,x,z);
    split(x,val-1,x,y);
    y = merge(fhq[y].l,fhq[y].r);
    root = merge(merge(x,y),z);
}
// 通过值获得排名
inline void getrank(int val){
    split(root,val-1,x,y);
    printf("%d\n",fhq[x].size+1);
    root = merge(x,y);
}
// 通过排名获得值
inline void getnum(int rank){
    int rt = root;
    while(rt){
        if(fhq[fhq[rt].l].size +1 == rank)  break;
        else if(fhq[fhq[rt].l].size >= rank)    rt = fhq[rt].l;
        else{
            rank -= fhq[fhq[rt].l].size + 1;
            rt = fhq[rt].r;
        }
    }
    printf("%d\n",fhq[rt].val);
}
// 前驱节点
inline void getpre(int val){
    split(root,val-1,x,y);
    int rt = x;
    while(fhq[rt].r)    rt = fhq[rt].r;
    printf("%d\n",fhq[rt].val);
    root = merge(x,y);
}
// 获得后继节点
inline void getnext(int val){
    split(root,val,x,y);
    int rt = y;
    while(fhq[rt].l)    rt = fhq[rt].l;
    printf("%d\n",fhq[rt].val);
    root = merge(x,y);
}
int main(){
    int n,opt,val;  scanf("%d",&n);
    while(n--){
        scanf("%d%d",&opt,&val);
        switch (opt)
        {
        case 1: insert(val);    break;
        case 2: del(val);       break;
        case 3: getrank(val);   break;
        case 4: getnum(val);    break;
        case 5: getpre(val);    break;
        case 6: getnext(val);   break;
        }
    }
    return 0;
}
```

## 匈牙利算法

二分图最大匹配

```cpp
int vN,uN;
int g[N][N];
int linker[N];
bool used[N];
bool dfs(int u){
    for(int v = 1;v<vN;v++)
        if(g[u][v] && !used[v]){
            used[v] = true;
            if(!linker[v] || dfs(linker[v])){
                linker[v] = u;
                return true;
            }
        }
    return false;
}
int hungary(){
    int res = 0;
    memset(linker,0,sizeof(linker));
    for(int u=1;u<uN;++u){
        memset(used,false,sizeof used);
        if(dfs(u))  res++;
    }
    return res;
}
```

## 点分治

统计树上路径信息

```cpp
#include<bits/stdc++.h>
#define ll long long
#define inf 1<<30
using namespace std;
const int N =1e5+5;
struct E{
    int v,w,nxt;
}e[N*2];

int vt,head[N];
int vis[N],size[N],maxv[N],dis[N];
int ans,root,ma;

int n,k,num;
void init(){
    vt = 0;    ans = 0;
    memset(head,0,sizeof(head));
    memset(vis,0,sizeof(vis));
}
void add(int u,int v,int w){
    e[vt].v = v;e[vt].w = w;e[vt].nxt = head[u];head[u] = vt++;
}

void dfs_size(int u,int f){
    size[u] = 1;
    maxv[u] = 0;
    for(int i=head[u];i;i=e[i].nxt){
        int v = e[i].v;
        if(v == f || vis[v])    continue;
        dfs_size(v,u);
        size[u] += size[v];
        maxv[u] = max(maxv[u],size[v]);
    }
}
void dfs_root(int r,int u,int f){
    maxv[u] = max(maxv[u],size[r]-size[u]);
    if(maxv[u] <ma){
        ma = maxv[u];
        root = u;
    }
    for(int i=head[u];i;i=e[i].nxt){
        int v = e[i].v;
        if(v == f || vis[v] )   continue;
        dfs_root(r,v,u);
    }
}
void dfs_dis(int u,int d,int f){
    dis[num++] = d;
    for(int i=head[u];i;i=e[i].nxt){
        int v = e[i].v;
        if(v == f || vis[v])    continue;
        dfs_dis(v,d+e[i].w,u);
    }
}
// 计算贡献
int calc(int u,int d){
    int res = 0;
    num = 0;
    dfs_dis(u,d,0);
    sort(dis,dis+num);
    int i=0,j=num-1;
    while(i<j){
        while(dis[i] + dis[j] > k && i<j)
            j--;
        res += j-i;
        i++;
    }
    return res;
}

void dfs(int u){
    ma = n;
    dfs_size(u,0);
    dfs_root(u,u,0);
    ans += calc(root,0);
    vis[root] = 1;
    for(int i=head[root];i;i=e[i].nxt){
        int v =e[i].v;
        if(!vis[v]){
            ans -= calc(v,e[i].w);
            dfs(v);
        }
    }
}

int main(){
    while(scanf("%d%d",&n,&k)){
        if(n==0 && k==0)    break;
        int u,v,w;
        init();
        for(int i=1;i<n;++i){
            scanf("%d%d%d",&u,&v,&w);
            add(u,v,w);
            add(v,u,w);
        }
        dfs(1);
        printf("%d\n",ans);
    }
    
}
```

## 线面关系(UPD:19.10.16)

```cpp
#include<cstdio>
#include<cmath>
#include<algorithm>
#define ll long long
using namespace std;
const int N = 1e3+10;
const double eps = 1e-8;
inline int sgn(double x){
    if(fabs(x)<eps) return 0;
    return x<0? -1:1;
}
// 防爆精度加
double add(double a,double b){
    if(fabs(a+b)<eps*(fabs(a)+fabs(b))) return 0;
    return a+b;
}

struct point{
    double x,y;
    point(double a=0,double b=0){
        x = a, y = b;
    }
    point operator -(const point &b)const{
        return point(x-b.x,y-b.y);
    }
    bool operator <(const point &b)const{
        return x<b.x-eps;
    }
    bool operator == (const point &b)const{
        return sgn(x-b.x) == 0 && sgn(y-b.y) == 0;
    }
    // b 为弧度制
    void transxy(double b){
        double tx = x,  ty = y;
        x = tx*cos(b) - ty*sin(b);
        y = tx*sin(b) + ty*cos(b);
    }
    double norm(){
        return sqrt(x*x+y*y);
    }
};
inline double det(const point &a,const point &b){
    return a.x*b.y - a.y*b.x;
}
inline double dot(const point &a,const point &b){
    return a.x*b.x+ a.y*b.y;
}
inline double dist(const point &a,const point &b){
    return (a-b).norm();
}
double area(point a,point b,point c){
    return det(b-a,c-b);
}
struct line{
    point s,e;
    line(){}
    line(point s,point e):s(s),e(e){}
};
inline bool parallel(line &l1,line &l2){
    return !sgn(det(l1.e - l1.s,l2.e-l2.s));
}
inline bool point_on_seg(point &p,line &a){
    return sgn(det(p-a.s,p-a.e))==0 && sgn(dot(p-a.s,p-a.e))<=0;
}
// 线段与线段相交
inline bool cross(line l1,line l2){
    line vec = line(l1.s,l1.e-l1.s);    // l1 的向量9
    if(parallel(l1,l2))     // 平行（重合），则点在线上才相交
        return point_on_seg(l1.s,l2) || point_on_seg(l1.e,l2)
            || point_on_seg(l2.s,l1) || point_on_seg(l2.e,l1);
    if(sgn(det(l2.s-vec.s,vec.e)*det(vec.e,l2.e-vec.s))==-1)
        return false; 
    if(sgn(det(l2.s-l1.s,l2.e-l1.s)*det(l2.s-l1.e,l2.e-l1.e))==1)
        return false;
    return true;
}
// 线段与直线相交
// 手写 uncheck
bool segcorseg(line l1,line l2){
    // 两端叉积符号不同，则在不同侧，说明相交   
    // = 0 表示端点在线上
    return sgn(det(l1.s - l2.s, l1.e - l2.s)) * sgn(det(l1.s - l2.e , l1.e - l2.e)) <= 0
    || sgn(det(l2.s - l1.s,l2.e - l1.s)) * sgn(det(l2.s - l1.e, l2.e - l1.e)) <= 0;
}
// 线段交点
point crosspoint(line l1,line l2){
    double a1 = det(l2.e-l2.s,l1.s-l2.s);
    double a2 = det(l2.e-l2.s,l1.e-l2.s);
    return point((l1.s.x*a2-l1.e.x*a1)/(a2-a1),(l1.s.y*a2-l1.e.y*a1)/(a2-a1));
}

// 极角排序
point start;
bool cmp(const point &a,const point &b){
    double x = det(a-start,b-start);
    if(sgn(x)==0)   return dist(start,a) < dist(start,b);
    else return x>0;
}
struct polygon{
    int n;
    point a[N];
    polygon(){}
    bool isTu(){
        // 凸多边形判断
        // 相邻三点构成三角形 叉积符号相同
        bool s[3];  // -1 0 1 是否存在， 
        s[0] = s[1] = s[2] = 0;
        for(int i=0;i<n;++i){
            s[sgn(area(a[i],a[(i+1)%n],a[(i+2)%n]))+1] = 1;
            if(s[0] && s[2])    return false;   // 有正又有负不行
        }
        return true;
    }
    bool inPoly(point p){
        // 点在多边形内
        // 先以a[0]为源点 进行极角排序
        start = a[0];
        sort(a,a+n,cmp);
        int cnt = 0 ;
        line ray,side;  // 点所在平行y轴射线，当前多边形的边
        ray.s = p;  ray.e.y = p.y;  ray.e.x = -1e9;
        for(int i=0;i<n;++i){
            side.s = a[i];
            side.e = a[(i+1)%n];
            if(point_on_seg(p,side))   return 1; // 如果点在边界上，也算在多边形内
            if(sgn(side.s.y-side.e.y)==0)   continue; // 平行 不计算
            if(point_on_seg(side.s,ray)){     // 起点在射线上
                if(sgn(side.s.y - side.e.y) > 0) cnt++; // 小于等于0 则不会在多边形内
            } 
            else if(point_on_seg(side.e,ray)){
                if(sgn(side.e.y - side.s.y) > 0) cnt++;
            }
            else if(cross(ray,side))   // 线段相交
                cnt++;          
        }
        return cnt%2; // 相交点数为奇数，在多边形内
    }
}poly;
```