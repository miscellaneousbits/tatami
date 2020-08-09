/*  cabbage.go -- Compute T(s) from Project Euler Problem 256
    Optimised from gplub.go December 29, 2019 by Eric Olson  */

package main

import ("fmt"; "os")

const (Pnum=1300; Smax=100000000; Fnum=10)
const (meow=5; bark=30)
type factors struct {
    s,smin,fmax,i,d int; p,n [Fnum]int
}
var (Tisn int; P [Pnum]int)

func doinit(){
    var Pn,in int
    isprime:=func(p int)(bool){
        var i int
        for i=0;i<in;i++ {
            if p%P[i]==0 { return false }
        }
        for i=in;P[i]*P[i]<=p;i++ {
            if p%P[i]==0 { return false }
        }
        in=i-1
        return true
    }
    var p int
    P[0]=2; P[1]=3; Pn=2; in=1
    for p=5;Pn<Pnum;p+=2 {
        if isprime(p) { P[Pn]=p; Pn++ }
    }
    if p<=Smax/p+1 {
        fmt.Printf("The maximum prime %d is too small!\n",p)
        os.Exit(1)
    }
    r:=1
    for i:=0;i<Fnum-1;i++ {
        if P[i]>Smax/r+1 { return }
        r*=P[i]
    }
    fmt.Printf("Distinct primes %d in factorisation too few!\n",Fnum)
    os.Exit(2)
}

func tfree(k,l int)(int){
    n:=l/k
    lmin:=(k+1)*n+2
    lmax:=(k-1)*(n+1)-2
    if lmin<=l && l<=lmax { return 1 }
    return 0
}
func ppow(p,n int)(int){
    if n==0 { return 1 }
    r:=1
    for {
        if n&1==1 { r*=p }
        n>>=1
        if n==0 { return r }
        p*=p
    }
}
func T(x *factors)(int){
    var z [Fnum]int
    for w:=0;w<Fnum;w++ { z[w]=0 }
    r:=0
    for {
        var i int
        for i=0;i<=x.fmax;i++ {
            if z[i]<x.n[i] {
                z[i]++; break
            }
            z[i]=0
        }
        if i>x.fmax { break }
        k:=1; l:=1
        for i=0;i<=x.fmax;i++ {
            k*=ppow(x.p[i],z[i])
        }
        l=x.s/k
        if k<=l { r+=tfree(k,l) }
    }
    return r
}

func sigma(x *factors)(int){
    r:=x.n[0]
    for i:=1;i<=x.fmax;i++ { r*=x.n[i]+1 }
    return r
}
func Twork(x *factors){
    s:=x.s; d:=x.d
    r:=sigma(x)
    if r>=Tisn {
        r=T(x)
        if r==Tisn&&s<x.smin { x.smin=s }
    }
    i:=x.i
    fmax:=x.fmax
    pmax:=x.smin/s+1
    p:=P[i]
    if p<=pmax {
        x.n[fmax]++; x.s=s*p; x.d=d+p
        Twork(x)
        x.n[fmax]--
    }
    fmax++
    x.n[fmax]=1
    for i++;i<Pnum;i++ {
        p=P[i]
        if p>pmax { break }
        x.p[fmax]=p; x.s=s*p; x.d=d+p
        x.i=i; x.fmax=fmax
        Twork(x)
    }
    x.n[fmax]=0
}

func Pwork(x factors,c chan int){
    s:=x.s; d:=x.d; echo:=0
    sync:=make(chan int,Pnum+2)
    Twrap:=func(){
        if x.d>bark {
            Twork(&x)
        } else if x.d<meow {
            Pwork(x,sync)
            t:=<-sync
            if t<x.smin { x.smin=t }
        } else {
            echo++
            go Pwork(x,sync)
        }
    }
    r:=sigma(&x)
    if r>=Tisn {
        r=T(&x)
        if r==Tisn&&s<x.smin { x.smin=s }
    }
    i:=x.i
    fmax:=x.fmax
    pmax:=x.smin/s+1
    p:=P[i]
    if p<=pmax {
        x.n[fmax]++; x.s=s*p; x.d=d+p
        Twrap()
        x.n[fmax]--
    }
    fmax++
    x.n[fmax]=1
    for i++;i<Pnum;i++ {
        p=P[i]
        if p>pmax { break }
        x.p[fmax]=p; x.s=s*p; x.d=d+p
        x.i=i; x.fmax=fmax
        Twrap()
    }
    x.n[fmax]=0
    for i=0;i<echo;i++ {
        t:=<-sync
        if t<x.smin { x.smin=t }
    }
    c<-x.smin
}

func Tinv(n int)(int){
    Tisn=n
    var x factors
    x.p[0]=P[0]; x.n[0]=1; x.i=0; x.s=2; x.d=2
    x.fmax=0; x.smin=Smax
    sync:=make(chan int,1)
    go Pwork(x,sync)
    smin:=<-sync
    if smin<Smax { return smin }
    return -1
}
func main(){
    n:=200
    doinit()
    fmt.Printf("P[%d]=%d\n",Pnum,P[Pnum-1])
    s:=Tinv(n)
    fmt.Printf("T(%d)=%d\n",s,n)
    os.Exit(0)
}

