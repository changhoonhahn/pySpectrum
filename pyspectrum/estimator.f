cc*******************************************************************
      subroutine ffting(dtl,N,Ngrid)
cc*******************************************************************
      integer, intent(in) :: N,Ngrid
      complex, intent(inout) :: dtl(Ngrid,Ngrid,Ngrid)
      integer*8 planf
      include '/usr/local/include/fftw3.f'
      write(*,*)'making plan'
      call sfftw_plan_dft_3d(planf,Ngrid,Ngrid,Ngrid,dtl,dtl, 
     &          FFTW_BACKWARD,FFTW_ESTIMATE)
      write(*,*)'doing fft'
      call sfftw_execute_dft(planf,dtl,dtl)
      write(*,*)'destroy plan'
      call sfftw_destroy_plan(planf)
      write(*,*)'doing fcomb'
      call fcomb(dtl,N,Ngrid)
      return 
      end 
cc*******************************************************************
      subroutine assign2(r,dtl,Np,Ngrid,kf_ks)
cc*******************************************************************
      integer, intent(in) :: Np,Ngrid
      real, intent(in) :: kf_ks 
      real, dimension(3,Np), intent(in) :: r
      real, dimension(2*Ngrid,Ngrid,Ngrid), intent(inout) :: dtl
c      ks = 2pi/Ngrid
c      kf = 2pi/Lbox 
c      do 1 iz=1,Ngrid
c       do 1 iy=1,Ngrid
c        do 1 ix=1,2*Ngrid
c1        dtl(ix,iy,iz)=0.

      do 2 i=1,Np
       rx=kf_ks*r(1,i)+1.
       ry=kf_ks*r(2,i)+1.
       rz=kf_ks*r(3,i)+1.
       tx=rx+0.5
       ty=ry+0.5
       tz=rz+0.5
       ixm1=int(rx)
       iym1=int(ry)
       izm1=int(rz)
       ixm2=2*mod(ixm1-2+Ngrid,Ngrid)+1
       ixp1=2*mod(ixm1,Ngrid)+1
       ixp2=2*mod(ixm1+1,Ngrid)+1
       hx=rx-ixm1
       ixm1=2*ixm1-1
       hx2=hx*hx
       hxm2=(1.-hx)**3
       hxm1=4.+(3.*hx-6.)*hx2
       hxp2=hx2*hx
       hxp1=6.-hxm2-hxm1-hxp2
c
       iym2=mod(iym1-2+Ngrid,Ngrid)+1
       iyp1=mod(iym1,Ngrid)+1
       iyp2=mod(iym1+1,Ngrid)+1
       hy=ry-iym1
       hy2=hy*hy
       hym2=(1.-hy)**3
       hym1=4.+(3.*hy-6.)*hy2
       hyp2=hy2*hy
       hyp1=6.-hym2-hym1-hyp2
c
       izm2=mod(izm1-2+Ngrid,Ngrid)+1
       izp1=mod(izm1,Ngrid)+1
       izp2=mod(izm1+1,Ngrid)+1
       hz=rz-izm1
       hz2=hz*hz
       hzm2=(1.-hz)**3
       hzm1=4.+(3.*hz-6.)*hz2
       hzp2=hz2*hz
       hzp1=6.-hzm2-hzm1-hzp2
c
       nxm1=int(tx)
       nym1=int(ty)
       nzm1=int(tz)
c
       gx=tx-nxm1
       nxm1=mod(nxm1-1,Ngrid)+1
       nxm2=2*mod(nxm1-2+Ngrid,Ngrid)+2
       nxp1=2*mod(nxm1,Ngrid)+2
       nxp2=2*mod(nxm1+1,Ngrid)+2
       nxm1=2*nxm1
       gx2=gx*gx
       gxm2=(1.-gx)**3
       gxm1=4.+(3.*gx-6.)*gx2
       gxp2=gx2*gx
       gxp1=6.-gxm2-gxm1-gxp2
c
       gy=ty-nym1
       nym1=mod(nym1-1,Ngrid)+1
       nym2=mod(nym1-2+Ngrid,Ngrid)+1
       nyp1=mod(nym1,Ngrid)+1
       nyp2=mod(nym1+1,Ngrid)+1
       gy2=gy*gy
       gym2=(1.-gy)**3
       gym1=4.+(3.*gy-6.)*gy2
       gyp2=gy2*gy
       gyp1=6.-gym2-gym1-gyp2
c
       gz=tz-nzm1
       nzm1=mod(nzm1-1,Ngrid)+1
       nzm2=mod(nzm1-2+Ngrid,Ngrid)+1
       nzp1=mod(nzm1,Ngrid)+1
       nzp2=mod(nzm1+1,Ngrid)+1
       gz2=gz*gz
       gzm2=(1.-gz)**3
       gzm1=4.+(3.*gz-6.)*gz2
       gzp2=gz2*gz
       gzp1=6.-gzm2-gzm1-gzp2
c
       dtl(ixm2,iym2,izm2)   = dtl(ixm2,iym2,izm2)+ hxm2*hym2 *hzm2
       dtl(ixm1,iym2,izm2)   = dtl(ixm1,iym2,izm2)+ hxm1*hym2 *hzm2
       dtl(ixp1,iym2,izm2)   = dtl(ixp1,iym2,izm2)+ hxp1*hym2 *hzm2
       dtl(ixp2,iym2,izm2)   = dtl(ixp2,iym2,izm2)+ hxp2*hym2 *hzm2
       dtl(ixm2,iym1,izm2)   = dtl(ixm2,iym1,izm2)+ hxm2*hym1 *hzm2
       dtl(ixm1,iym1,izm2)   = dtl(ixm1,iym1,izm2)+ hxm1*hym1 *hzm2
       dtl(ixp1,iym1,izm2)   = dtl(ixp1,iym1,izm2)+ hxp1*hym1 *hzm2
       dtl(ixp2,iym1,izm2)   = dtl(ixp2,iym1,izm2)+ hxp2*hym1 *hzm2
       dtl(ixm2,iyp1,izm2)   = dtl(ixm2,iyp1,izm2)+ hxm2*hyp1 *hzm2
       dtl(ixm1,iyp1,izm2)   = dtl(ixm1,iyp1,izm2)+ hxm1*hyp1 *hzm2
       dtl(ixp1,iyp1,izm2)   = dtl(ixp1,iyp1,izm2)+ hxp1*hyp1 *hzm2
       dtl(ixp2,iyp1,izm2)   = dtl(ixp2,iyp1,izm2)+ hxp2*hyp1 *hzm2
       dtl(ixm2,iyp2,izm2)   = dtl(ixm2,iyp2,izm2)+ hxm2*hyp2 *hzm2
       dtl(ixm1,iyp2,izm2)   = dtl(ixm1,iyp2,izm2)+ hxm1*hyp2 *hzm2
       dtl(ixp1,iyp2,izm2)   = dtl(ixp1,iyp2,izm2)+ hxp1*hyp2 *hzm2
       dtl(ixp2,iyp2,izm2)   = dtl(ixp2,iyp2,izm2)+ hxp2*hyp2 *hzm2
       dtl(ixm2,iym2,izm1)   = dtl(ixm2,iym2,izm1)+ hxm2*hym2 *hzm1
       dtl(ixm1,iym2,izm1)   = dtl(ixm1,iym2,izm1)+ hxm1*hym2 *hzm1
       dtl(ixp1,iym2,izm1)   = dtl(ixp1,iym2,izm1)+ hxp1*hym2 *hzm1
       dtl(ixp2,iym2,izm1)   = dtl(ixp2,iym2,izm1)+ hxp2*hym2 *hzm1
       dtl(ixm2,iym1,izm1)   = dtl(ixm2,iym1,izm1)+ hxm2*hym1 *hzm1
       dtl(ixm1,iym1,izm1)   = dtl(ixm1,iym1,izm1)+ hxm1*hym1 *hzm1
       dtl(ixp1,iym1,izm1)   = dtl(ixp1,iym1,izm1)+ hxp1*hym1 *hzm1
       dtl(ixp2,iym1,izm1)   = dtl(ixp2,iym1,izm1)+ hxp2*hym1 *hzm1
       dtl(ixm2,iyp1,izm1)   = dtl(ixm2,iyp1,izm1)+ hxm2*hyp1 *hzm1
       dtl(ixm1,iyp1,izm1)   = dtl(ixm1,iyp1,izm1)+ hxm1*hyp1 *hzm1
       dtl(ixp1,iyp1,izm1)   = dtl(ixp1,iyp1,izm1)+ hxp1*hyp1 *hzm1
       dtl(ixp2,iyp1,izm1)   = dtl(ixp2,iyp1,izm1)+ hxp2*hyp1 *hzm1
       dtl(ixm2,iyp2,izm1)   = dtl(ixm2,iyp2,izm1)+ hxm2*hyp2 *hzm1
       dtl(ixm1,iyp2,izm1)   = dtl(ixm1,iyp2,izm1)+ hxm1*hyp2 *hzm1
       dtl(ixp1,iyp2,izm1)   = dtl(ixp1,iyp2,izm1)+ hxp1*hyp2 *hzm1
       dtl(ixp2,iyp2,izm1)   = dtl(ixp2,iyp2,izm1)+ hxp2*hyp2 *hzm1
       dtl(ixm2,iym2,izp1)   = dtl(ixm2,iym2,izp1)+ hxm2*hym2 *hzp1
       dtl(ixm1,iym2,izp1)   = dtl(ixm1,iym2,izp1)+ hxm1*hym2 *hzp1
       dtl(ixp1,iym2,izp1)   = dtl(ixp1,iym2,izp1)+ hxp1*hym2 *hzp1
       dtl(ixp2,iym2,izp1)   = dtl(ixp2,iym2,izp1)+ hxp2*hym2 *hzp1
       dtl(ixm2,iym1,izp1)   = dtl(ixm2,iym1,izp1)+ hxm2*hym1 *hzp1
       dtl(ixm1,iym1,izp1)   = dtl(ixm1,iym1,izp1)+ hxm1*hym1 *hzp1
       dtl(ixp1,iym1,izp1)   = dtl(ixp1,iym1,izp1)+ hxp1*hym1 *hzp1
       dtl(ixp2,iym1,izp1)   = dtl(ixp2,iym1,izp1)+ hxp2*hym1 *hzp1
       dtl(ixm2,iyp1,izp1)   = dtl(ixm2,iyp1,izp1)+ hxm2*hyp1 *hzp1
       dtl(ixm1,iyp1,izp1)   = dtl(ixm1,iyp1,izp1)+ hxm1*hyp1 *hzp1
       dtl(ixp1,iyp1,izp1)   = dtl(ixp1,iyp1,izp1)+ hxp1*hyp1 *hzp1
       dtl(ixp2,iyp1,izp1)   = dtl(ixp2,iyp1,izp1)+ hxp2*hyp1 *hzp1
       dtl(ixm2,iyp2,izp1)   = dtl(ixm2,iyp2,izp1)+ hxm2*hyp2 *hzp1
       dtl(ixm1,iyp2,izp1)   = dtl(ixm1,iyp2,izp1)+ hxm1*hyp2 *hzp1
       dtl(ixp1,iyp2,izp1)   = dtl(ixp1,iyp2,izp1)+ hxp1*hyp2 *hzp1
       dtl(ixp2,iyp2,izp1)   = dtl(ixp2,iyp2,izp1)+ hxp2*hyp2 *hzp1
       dtl(ixm2,iym2,izp2)   = dtl(ixm2,iym2,izp2)+ hxm2*hym2 *hzp2
       dtl(ixm1,iym2,izp2)   = dtl(ixm1,iym2,izp2)+ hxm1*hym2 *hzp2
       dtl(ixp1,iym2,izp2)   = dtl(ixp1,iym2,izp2)+ hxp1*hym2 *hzp2
       dtl(ixp2,iym2,izp2)   = dtl(ixp2,iym2,izp2)+ hxp2*hym2 *hzp2
       dtl(ixm2,iym1,izp2)   = dtl(ixm2,iym1,izp2)+ hxm2*hym1 *hzp2
       dtl(ixm1,iym1,izp2)   = dtl(ixm1,iym1,izp2)+ hxm1*hym1 *hzp2
       dtl(ixp1,iym1,izp2)   = dtl(ixp1,iym1,izp2)+ hxp1*hym1 *hzp2
       dtl(ixp2,iym1,izp2)   = dtl(ixp2,iym1,izp2)+ hxp2*hym1 *hzp2
       dtl(ixm2,iyp1,izp2)   = dtl(ixm2,iyp1,izp2)+ hxm2*hyp1 *hzp2
       dtl(ixm1,iyp1,izp2)   = dtl(ixm1,iyp1,izp2)+ hxm1*hyp1 *hzp2
       dtl(ixp1,iyp1,izp2)   = dtl(ixp1,iyp1,izp2)+ hxp1*hyp1 *hzp2
       dtl(ixp2,iyp1,izp2)   = dtl(ixp2,iyp1,izp2)+ hxp2*hyp1 *hzp2
       dtl(ixm2,iyp2,izp2)   = dtl(ixm2,iyp2,izp2)+ hxm2*hyp2 *hzp2
       dtl(ixm1,iyp2,izp2)   = dtl(ixm1,iyp2,izp2)+ hxm1*hyp2 *hzp2
       dtl(ixp1,iyp2,izp2)   = dtl(ixp1,iyp2,izp2)+ hxp1*hyp2 *hzp2
       dtl(ixp2,iyp2,izp2)   = dtl(ixp2,iyp2,izp2)+ hxp2*hyp2 *hzp2
c
       dtl(nxm2,nym2,nzm2)   = dtl(nxm2,nym2,nzm2)+ gxm2*gym2 *gzm2
       dtl(nxm1,nym2,nzm2)   = dtl(nxm1,nym2,nzm2)+ gxm1*gym2 *gzm2
       dtl(nxp1,nym2,nzm2)   = dtl(nxp1,nym2,nzm2)+ gxp1*gym2 *gzm2
       dtl(nxp2,nym2,nzm2)   = dtl(nxp2,nym2,nzm2)+ gxp2*gym2 *gzm2
       dtl(nxm2,nym1,nzm2)   = dtl(nxm2,nym1,nzm2)+ gxm2*gym1 *gzm2
       dtl(nxm1,nym1,nzm2)   = dtl(nxm1,nym1,nzm2)+ gxm1*gym1 *gzm2
       dtl(nxp1,nym1,nzm2)   = dtl(nxp1,nym1,nzm2)+ gxp1*gym1 *gzm2
       dtl(nxp2,nym1,nzm2)   = dtl(nxp2,nym1,nzm2)+ gxp2*gym1 *gzm2
       dtl(nxm2,nyp1,nzm2)   = dtl(nxm2,nyp1,nzm2)+ gxm2*gyp1 *gzm2
       dtl(nxm1,nyp1,nzm2)   = dtl(nxm1,nyp1,nzm2)+ gxm1*gyp1 *gzm2
       dtl(nxp1,nyp1,nzm2)   = dtl(nxp1,nyp1,nzm2)+ gxp1*gyp1 *gzm2
       dtl(nxp2,nyp1,nzm2)   = dtl(nxp2,nyp1,nzm2)+ gxp2*gyp1 *gzm2
       dtl(nxm2,nyp2,nzm2)   = dtl(nxm2,nyp2,nzm2)+ gxm2*gyp2 *gzm2
       dtl(nxm1,nyp2,nzm2)   = dtl(nxm1,nyp2,nzm2)+ gxm1*gyp2 *gzm2
       dtl(nxp1,nyp2,nzm2)   = dtl(nxp1,nyp2,nzm2)+ gxp1*gyp2 *gzm2
       dtl(nxp2,nyp2,nzm2)   = dtl(nxp2,nyp2,nzm2)+ gxp2*gyp2 *gzm2
       dtl(nxm2,nym2,nzm1)   = dtl(nxm2,nym2,nzm1)+ gxm2*gym2 *gzm1
       dtl(nxm1,nym2,nzm1)   = dtl(nxm1,nym2,nzm1)+ gxm1*gym2 *gzm1
       dtl(nxp1,nym2,nzm1)   = dtl(nxp1,nym2,nzm1)+ gxp1*gym2 *gzm1
       dtl(nxp2,nym2,nzm1)   = dtl(nxp2,nym2,nzm1)+ gxp2*gym2 *gzm1
       dtl(nxm2,nym1,nzm1)   = dtl(nxm2,nym1,nzm1)+ gxm2*gym1 *gzm1
       dtl(nxm1,nym1,nzm1)   = dtl(nxm1,nym1,nzm1)+ gxm1*gym1 *gzm1
       dtl(nxp1,nym1,nzm1)   = dtl(nxp1,nym1,nzm1)+ gxp1*gym1 *gzm1
       dtl(nxp2,nym1,nzm1)   = dtl(nxp2,nym1,nzm1)+ gxp2*gym1 *gzm1
       dtl(nxm2,nyp1,nzm1)   = dtl(nxm2,nyp1,nzm1)+ gxm2*gyp1 *gzm1
       dtl(nxm1,nyp1,nzm1)   = dtl(nxm1,nyp1,nzm1)+ gxm1*gyp1 *gzm1
       dtl(nxp1,nyp1,nzm1)   = dtl(nxp1,nyp1,nzm1)+ gxp1*gyp1 *gzm1
       dtl(nxp2,nyp1,nzm1)   = dtl(nxp2,nyp1,nzm1)+ gxp2*gyp1 *gzm1
       dtl(nxm2,nyp2,nzm1)   = dtl(nxm2,nyp2,nzm1)+ gxm2*gyp2 *gzm1
       dtl(nxm1,nyp2,nzm1)   = dtl(nxm1,nyp2,nzm1)+ gxm1*gyp2 *gzm1
       dtl(nxp1,nyp2,nzm1)   = dtl(nxp1,nyp2,nzm1)+ gxp1*gyp2 *gzm1
       dtl(nxp2,nyp2,nzm1)   = dtl(nxp2,nyp2,nzm1)+ gxp2*gyp2 *gzm1
       dtl(nxm2,nym2,nzp1)   = dtl(nxm2,nym2,nzp1)+ gxm2*gym2 *gzp1
       dtl(nxm1,nym2,nzp1)   = dtl(nxm1,nym2,nzp1)+ gxm1*gym2 *gzp1
       dtl(nxp1,nym2,nzp1)   = dtl(nxp1,nym2,nzp1)+ gxp1*gym2 *gzp1
       dtl(nxp2,nym2,nzp1)   = dtl(nxp2,nym2,nzp1)+ gxp2*gym2 *gzp1
       dtl(nxm2,nym1,nzp1)   = dtl(nxm2,nym1,nzp1)+ gxm2*gym1 *gzp1
       dtl(nxm1,nym1,nzp1)   = dtl(nxm1,nym1,nzp1)+ gxm1*gym1 *gzp1
       dtl(nxp1,nym1,nzp1)   = dtl(nxp1,nym1,nzp1)+ gxp1*gym1 *gzp1
       dtl(nxp2,nym1,nzp1)   = dtl(nxp2,nym1,nzp1)+ gxp2*gym1 *gzp1
       dtl(nxm2,nyp1,nzp1)   = dtl(nxm2,nyp1,nzp1)+ gxm2*gyp1 *gzp1
       dtl(nxm1,nyp1,nzp1)   = dtl(nxm1,nyp1,nzp1)+ gxm1*gyp1 *gzp1
       dtl(nxp1,nyp1,nzp1)   = dtl(nxp1,nyp1,nzp1)+ gxp1*gyp1 *gzp1
       dtl(nxp2,nyp1,nzp1)   = dtl(nxp2,nyp1,nzp1)+ gxp2*gyp1 *gzp1
       dtl(nxm2,nyp2,nzp1)   = dtl(nxm2,nyp2,nzp1)+ gxm2*gyp2 *gzp1
       dtl(nxm1,nyp2,nzp1)   = dtl(nxm1,nyp2,nzp1)+ gxm1*gyp2 *gzp1
       dtl(nxp1,nyp2,nzp1)   = dtl(nxp1,nyp2,nzp1)+ gxp1*gyp2 *gzp1
       dtl(nxp2,nyp2,nzp1)   = dtl(nxp2,nyp2,nzp1)+ gxp2*gyp2 *gzp1
       dtl(nxm2,nym2,nzp2)   = dtl(nxm2,nym2,nzp2)+ gxm2*gym2 *gzp2
       dtl(nxm1,nym2,nzp2)   = dtl(nxm1,nym2,nzp2)+ gxm1*gym2 *gzp2
       dtl(nxp1,nym2,nzp2)   = dtl(nxp1,nym2,nzp2)+ gxp1*gym2 *gzp2
       dtl(nxp2,nym2,nzp2)   = dtl(nxp2,nym2,nzp2)+ gxp2*gym2 *gzp2
       dtl(nxm2,nym1,nzp2)   = dtl(nxm2,nym1,nzp2)+ gxm2*gym1 *gzp2
       dtl(nxm1,nym1,nzp2)   = dtl(nxm1,nym1,nzp2)+ gxm1*gym1 *gzp2
       dtl(nxp1,nym1,nzp2)   = dtl(nxp1,nym1,nzp2)+ gxp1*gym1 *gzp2
       dtl(nxp2,nym1,nzp2)   = dtl(nxp2,nym1,nzp2)+ gxp2*gym1 *gzp2
       dtl(nxm2,nyp1,nzp2)   = dtl(nxm2,nyp1,nzp2)+ gxm2*gyp1 *gzp2
       dtl(nxm1,nyp1,nzp2)   = dtl(nxm1,nyp1,nzp2)+ gxm1*gyp1 *gzp2
       dtl(nxp1,nyp1,nzp2)   = dtl(nxp1,nyp1,nzp2)+ gxp1*gyp1 *gzp2
       dtl(nxp2,nyp1,nzp2)   = dtl(nxp2,nyp1,nzp2)+ gxp2*gyp1 *gzp2
       dtl(nxm2,nyp2,nzp2)   = dtl(nxm2,nyp2,nzp2)+ gxm2*gyp2 *gzp2
       dtl(nxm1,nyp2,nzp2)   = dtl(nxm1,nyp2,nzp2)+ gxm1*gyp2 *gzp2
       dtl(nxp1,nyp2,nzp2)   = dtl(nxp1,nyp2,nzp2)+ gxp1*gyp2 *gzp2
       dtl(nxp2,nyp2,nzp2)   = dtl(nxp2,nyp2,nzp2)+ gxp2*gyp2 *gzp2
2     continue
c
      return
      end
cc*******************************************************************
      subroutine fcomb(dcl,N,Ngrid)
cc*******************************************************************
      integer, intent(in) :: N,Ngrid
      complex, intent(inout) :: dcl(Ngrid,Ngrid,Ngrid)
      parameter(tpi=6.283185307d0)
      real*8 tpiL,piL
      complex*16 recx,recy,recz,xrec,yrec,zrec
      complex c1,ci,c000,c001,c010,c011,cma,cmb,cmc,cmd
Cf2py intent(in) n
Cf2py intent(in) ngrid
Cf2py intent(inout) dcl 
      cf=1./(6.**3*4.*N)

      Lnyq=Ngrid/2+1
      tpiL=tpi/float(Ngrid)
      piL=-tpiL/2.
      recx=cmplx(dcos(piL),dsin(piL))
      
      c1=cmplx(1.,0.)
      ci=cmplx(0.,1.)
      zrec=c1
      do 301 iz=1,Lnyq
       icz=mod(Ngrid-iz+1,Ngrid)+1
       rkz=tpiL*(iz-1)
       Wkz=1.
       if(rkz.ne.0.)Wkz=(sin(rkz/2.)/(rkz/2.))**4
       yrec=c1
       do 302 iy=1,Lnyq
        icy=mod(Ngrid-iy+1,Ngrid)+1
        rky=tpiL*(iy-1)
        Wky=1.
        if(rky.ne.0.)Wky=(sin(rky/2.)/(rky/2.))**4
        xrec=c1
        do 303 ix=1,Lnyq
         icx=mod(Ngrid-ix+1,Ngrid)+1
         rkx=tpiL*(ix-1)
         Wkx=1.
         if(rkx.ne.0.)Wkx=(sin(rkx/2.)/(rkx/2.))**4
         cfac=cf/(Wkx*Wky*Wkz)
c
         cma=ci*xrec*yrec*zrec
         cmb=ci*xrec*yrec*conjg(zrec)
         cmc=ci*xrec*conjg(yrec)*zrec
         cmd=ci*xrec*conjg(yrec*zrec)
c
         c000=dcl(ix,iy ,iz )*(c1-cma)+conjg(dcl(icx,icy,icz))*(c1+cma)
         c001=dcl(ix,iy ,icz)*(c1-cmb)+conjg(dcl(icx,icy,iz ))*(c1+cmb)
         c010=dcl(ix,icy,iz )*(c1-cmc)+conjg(dcl(icx,iy ,icz))*(c1+cmc)
         c011=dcl(ix,icy,icz)*(c1-cmd)+conjg(dcl(icx,iy ,iz ))*(c1+cmd)
c
c
         dcl(ix,iy ,iz )=c000*cfac
         dcl(ix,iy ,icz)=c001*cfac
         dcl(ix,icy,iz )=c010*cfac
         dcl(ix,icy,icz)=c011*cfac
         dcl(icx,iy ,iz )=conjg(dcl(ix,icy,icz))
         dcl(icx,iy ,icz)=conjg(dcl(ix,icy,iz ))
         dcl(icx,icy,iz )=conjg(dcl(ix,iy ,icz))
         dcl(icx,icy,icz)=conjg(dcl(ix,iy ,iz ))
c
         xrec=xrec*recx
303     continue
        yrec=yrec*recy
302    continue
       zrec=zrec*recz
301   continue
c
      return
      end

