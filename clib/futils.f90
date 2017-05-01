subroutine flncoshd(x,y)
    implicit none
    real*8,parameter :: log2_=log(2D0)
    real*8,intent(in) :: x
    real*8,intent(out) :: y
    real*8 :: absx
    !f2py intent(in) :: x
    !f2py intent(out) :: y
    absx=abs(x)
    if(x>12) then
        y=x-log2_
    else
        y=log(cosh(x))
    endif
end subroutine flncoshd

subroutine flncoshc(x,y)
    implicit none
    complex*16,intent(in) :: x
    complex*16,intent(out) :: y
    real*8 :: xr,xi,yr
    !f2py intent(in) :: x
    !f2py intent(out) :: y
    xr=real(x)
    xi=imag(x)
    call flncoshd(xr,yr)
    y=yr+log(dcmplx(cos(xi),tanh(xr)*sin(xi)))
end subroutine flncoshc

subroutine fpop_nogroup(config,flips,wt,a,theta,ntheta,pratio,nf,nin,nhid,fh_id)
    implicit none
    integer,intent(in) :: nf,nin,nhid,fh_id
    integer,intent(in) :: config(nin),flips(nf)
    complex*16,intent(in) :: wt(nhid,nin),a(nin),theta(nhid)
    complex*16,intent(out) :: ntheta(nhid),pratio
    integer :: iflip,ci,i
    complex*16 :: y1,y0

    !f2py intent(in) :: nf,nin,nhid,config,flips,wt,a,theta,fh_id
    !f2py intent(out) :: ntheta,pratio

    ntheta=theta
    pratio=dcmplx(0D0,0D0)
    do i=1,nf
        iflip=flips(i)+1
        ci=config(iflip)
        ntheta=ntheta-2*ci*wt(:,iflip)
        pratio=pratio-2*ci*a(iflip)
    enddo
    if(fh_id==0) then
        do i=1,nhid
            y1=log(cosh(ntheta(i)))
            y0=log(cosh(theta(i)))
            pratio=pratio+y1-y0
        enddo
    elseif(fh_id==1) then
        do i=1,nhid
            y1=log(sinh(ntheta(i)))
            y0=log(sinh(theta(i)))
            pratio=pratio+y1-y0
        enddo
    elseif(fh_id==2)then
        do i=1,nhid
            call flncoshc(ntheta(i),y1)
            call flncoshc(theta(i),y0)
            pratio=pratio+y1-y0
        enddo
    else
        print*,'Error, invalid fh_id!'
        stop 2
    endif
    pratio=exp(pratio)
end subroutine fpop_nogroup
