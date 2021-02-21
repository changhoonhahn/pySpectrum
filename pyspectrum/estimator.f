cc*******************************************************************
      subroutine bk_counts(coun,nside,step,ncut,nmax)
cc*******************************************************************
      integer, intent(in) :: nmax,nside,ncut
      real, intent(in) :: step 
      real*8, intent(inout) :: coun(nmax,nmax,nmax)
      integer*8 planf
      integer Dim,nsideD
      parameter(Dim=3) 
      integer i,j,l,m,n,ncuts
      integer, allocatable :: indx(:)
      real di,dj,dl,dist
      real*8 sumn
      integer nk(0:2*nside), nbk(0:2*nside+1)
      real, allocatable :: normk(:,:),norm1(:,:)
c      include '/usr/local/include/fftw3.f'
c      include '~/project/pySpectrum/dat/fftw3.f'
      include 'dat/fftw3.f'

      ncuts=ncut/int(step)
      nsideD=nside**Dim
      allocate(indx(nsideD))
      do i=0,2*nside
         nk(i)=0
      enddo

c     Find modes of amplitude |k|
      write(*,*) 'Find modes of amplitude |k|'
      do i = 0, nside-1
         do j = 0, nside-1
            do l = 0, nside-1
               di = float(min(i,nside-i))
               dj = float(min(j,nside-j))
               dl = float(min(l,nside-l))
               dist = sqrt(di*di + dj*dj +dl*dl)
               nk(int(dist/step+0.5)) = nk(int(dist/step+0.5)) + 1 
            enddo
         enddo
      enddo
      
      nbk(0) = 0 
      do i = 0,2*nside
         nbk(i+1) = nbk(i) + nk(i)
         nk(i) = 0 
      enddo 
      
c     Save coordinates  
      write(*,*) 'Save coordinates'
      m = 0
      do i = 0, nside -1 
         do j = 0, nside -1
            do l = 0, nside -1
               di = float(min(i,nside-i))
               dj = float(min(j,nside-j))
               dl = float(min(l,nside-l))
               dist = sqrt(di*di + dj*dj +dl*dl)
               nk(int(dist/step+0.5)) = nk(int(dist/step+0.5)) + 1  
               n = nbk(int(dist/step+0.5)) + nk(int(dist/step+0.5))
               m = m+ 1
               indx(n) = m 
            enddo 
         enddo 
      enddo 
      
c     calculate k maps 
      write(*,*) 'Calculate k maps'
      allocate(norm1(2,nsideD),normk(nsideD,nmax))
      write(*,*) 'making plan'
      call sfftw_plan_dft_3d(planf,nside,nside,nside,norm1,norm1,
     &                       FFTW_FORWARD, FFTW_ESTIMATE)

      write(*,*) 'executing plan'
      do i = Ncut/int(step), nmax
         do n = 1, nsideD 
            norm1(1,n) = 0.
            norm1(2,n) = 0.
         enddo 
         do n = nbk(i)+1, nbk(i+1)
            norm1(1,indx(n)) = 1.
         enddo 
         call sfftw_execute_dft(planf,norm1,norm1)
c      call fftwnd_f77_one(planf,norm1,0)
         do n = 1, nsideD
            normk(n,i) = norm1(1,n)
         enddo
      enddo
      call sfftw_destroy_plan(planf)

      write(*,*) 'Final sum'
      do l = ncuts, nmax !keep ffts in outer loop
         do j = ncuts, l
            do i = max(ncuts,l-j), j
               sumn = 0.d0 
               do n = 1, nsideD
        sumn = sumn + dble(normk(n,i))*dble(normk(n,j))*dble(normk(n,l))
               enddo
               coun(i,j,l)=sumn
             enddo
         enddo 
      enddo          
      return 
      end
cc*******************************************************************
      subroutine pk_periodic(dtl,avgk,avgP,Ngrid,Nbin)
cc*******************************************************************
      integer, intent(in) :: Ngrid,Nbin
      complex, intent(inout) :: dtl(Ngrid/2+1,Ngrid,Ngrid)
      real*8, intent(inout) :: avgk(Nbin),avgP(Nbin)
      real*8 co(Nbin)
      real*8 shot
      real akx,aky,akz,phys_nyq
      complex ct

      do 10 i=1,Nbin
         avgk(i)=0.d0
         avgP(i)=0.d0
         co(i)=0.d0
 10   continue

      do 100 iz=1,Ngrid
         icz=mod(Ngrid+1-iz,Ngrid)+1
         rkz=akz*float(mod(iz+Ngrid/2-2,Ngrid)-Ngrid/2+1) 
         do 100 iy=1,Ngrid
            icy=mod(Ngrid+1-iy,Ngrid)+1
            rky=aky*float(mod(iy+Ngrid/2-2,Ngrid)-Ngrid/2+1)
            do 100 ix=1,Ngrid
               icx=mod(Ngrid+1-ix,Ngrid)+1
               rkx=akx*float(mod(ix+Ngrid/2-2,Ngrid)-Ngrid/2+1)
               rk=sqrt(rkx**2+rky**2+rkz**2)
               imk=nint(Nbin*rk/phys_nyq)  

               if(imk.le.Nbin .and. imk.ne.0)then
                  co(imk)=co(imk)+1.d0
                  if (ix.le.Ngrid/2+1) then 
                     ct=dtl(ix,iy,iz)
                  else !use cc
                     ct=dtl(icx,icy,icz)
                  endif
                  pk=(cabs(ct))**2
                  avgk(imk)=avgk(imk)+dble(rk)
                  avgP(imk)=avgP(imk)+dble(pk)
               end if
 100        continue
c******************************************************************
      shot=1.d0/dble(akx*aky*akz)/dble(Npar)
      do 110 Ibin=1,Nbin
         if(co(Ibin).gt.0.)then
            avgk(Ibin)=avgk(Ibin)/co(Ibin)  
            avgP(Ibin)=avgP(Ibin)/co(Ibin)/dble(akx*aky*akz)
         endif
 110  continue
      return 
      end
cc*******************************************************************
      subroutine pk_pbox_rsd(dtl,k,p0,p2,p4,nk,km,mk,pkm,nkm,
     &          irsd,Lbox,Nbin,Nmu,Ngrid)    
cc*******************************************************************
      integer, intent(in) :: Lbox,irsd,Nbin,Nmu,Ngrid
      complex, dimension(Ngrid/2+1,Ngrid,Ngrid), intent(in) :: dtl
      real*8, dimension(Nbin), intent(out) :: k,nk,p0,p2,p4
      real*8, dimension(Nbin,Nmu), intent(out) :: km,mk,pkm,nkm
      real pi,tpi,pk,rkx,rky,rkz,rk
      integer i,ibin,icx,icy,icz,imk,imu,imubin,ix,iy,iz,j
      parameter(pi=3.141592654,tpi=2.*pi)
      real*8 mu,Le2,Le4
      real mubin,thetaobs,phiobs
      real kf,cot1,sit1,cp,sp,cc
      complex ct
      kf=tpi/Lbox !fundamental mode

      mubin=1./real(Nmu) 
      if (irsd.eq.0) then
         thetaobs= 0.5*pi
         phiobs=0. 
      elseif (irsd.eq.1) then
         thetaobs=0.5*pi
         phiobs=0.5*pi
      elseif (irsd.eq.2) then
         thetaobs=0. 
         phiobs=0. 
      endif

      do 10 i=1,Nbin
         k(i)=0.d0
         p0(i)=0.d0
         p2(i)=0.d0
         p4(i)=0.d0
         nk(i)=0.d0
         do 10 j=1,Nmu
             km(i,j)=0.d0
             mk(i,j)=0.d0
             pkm(i,j)=0.d0
             nkm(i,j)=0.d0
 10   continue

      do 100 iz=1,Ngrid
         icz=mod(Ngrid+1-iz,Ngrid)+1
         rkz=real(mod(iz+Ngrid/2-2,Ngrid)-Ngrid/2+1) 
         do 100 iy=1,Ngrid
            icy=mod(Ngrid+1-iy,Ngrid)+1
            rky=real(mod(iy+Ngrid/2-2,Ngrid)-Ngrid/2+1)
            do 100 ix=1,Ngrid
               icx=mod(Ngrid+1-ix,Ngrid)+1
               rkx=real(mod(ix+Ngrid/2-2,Ngrid)-Ngrid/2+1)

               rk=sqrt(rkx**2+rky**2+rkz**2)
               imk=nint(Nbin*rk/real(Ngrid/2))  

               if(imk.le.Nbin .and. imk.ne.0)then
                  cot1=rkz/rk
                  sit1=sqrt(1.-cot1*cot1)
                  if (sit1.gt.0.) then
                     cp=rkx/(rk*sit1)
                     sp=rky/(rk*sit1)
                     cc=sin(phiobs)*sp+cos(phiobs)*cp
                  else
                     cc=0.
                  endif
                  mu=dble(cos(thetaobs)*cot1+sin(thetaobs)*sit1*cc)!mu 
                  imu=int((abs(mu)+dble(mubin))/dble(mubin))

                  Le2=-5.d-1+1.5d0*mu**2
                  Le4=3.75d-1-3.75d0*mu**2+4.375d0*mu**4

                  nk(imk)=nk(imk)+1.d0
                  if (ix.le.Ngrid/2+1) then 
                     ct=dtl(ix,iy,iz)
                  else !use cc
                     ct=dtl(icx,icy,icz)
                  endif
                  pk=(cabs(ct))**2
                  k(imk)=k(imk)+dble(kf*rk)
                  p0(imk)=p0(imk)+dble(pk)
                  p2(imk)=p2(imk)+dble(pk)*5.d0*Le2
                  p4(imk)=p4(imk)+dble(pk)*9.d0*Le4

                  if(imu.le.Nmu .and. imu.gt.0)then 
                      nkm(imk,imu)=nkm(imk,imu)+1.d0
                      km(imk,imu)=km(imk,imu)+dble(kf*rk)
                      mk(imk,imu)=mk(imk,imu)+dble(abs(mu))
                      pkm(imk,imu)=pkm(imk,imu)+dble(pk)
                  endif 
               endif
 100        continue
c******************************************************************
      do 20 Ibin=1,Nbin
        if(nk(Ibin).gt.0)then 
            k(Ibin)=k(Ibin)/nk(Ibin)  
            p0(Ibin)=p0(Ibin)/nk(Ibin)/dble(kf**3)
            p2(Ibin)=p2(Ibin)/nk(Ibin)/dble(kf**3)
            p4(Ibin)=p4(Ibin)/nk(Ibin)/dble(kf**3)
        endif 
 20   continue
      do 110 Ibin=1,Nbin
         do 110 Imubin=1,Nmu
            if(nkm(Ibin,Imubin).gt.0)then 
                km(Ibin,Imubin)=km(Ibin,Imubin)/nkm(Ibin,Imubin)
                mk(Ibin,Imubin)=mk(Ibin,Imubin)/nkm(Ibin,Imubin)
                Pkm(Ibin,Imubin)=Pkm(Ibin,Imubin)/nkm(Ibin,Imubin)
     &              /dble(kf**3)
            endif 
 110  continue
      return 
      end
cc*******************************************************************
      subroutine ffting(dtl,N,Ngrid)
cc*******************************************************************
      integer, intent(in) :: N,Ngrid
      complex, intent(inout) :: dtl(Ngrid,Ngrid,Ngrid)
      integer*8 planf
c      include '/usr/local/include/fftw3.f'
c      include '~/project/pySpectrum/dat/fftw3.f'
      include 'dat/fftw3.f'
      write(*,*)'making plan'
      call sfftw_plan_dft_3d(planf,Ngrid,Ngrid,Ngrid,dtl,dtl, 
     &          FFTW_BACKWARD,FFTW_ESTIMATE)
      write(*,*)'doing fft'
      call sfftw_execute_dft(planf,dtl,dtl)
      write(*,*)'destroy plan'
      call sfftw_destroy_plan(planf)
      return 
      end 
cc*******************************************************************
      subroutine assign_quad(r,w,dtl,Np,Ngrid,kf_ks,ia,ib,ic,id)
cc*******************************************************************
      integer, intent(in) :: Np,Ngrid,ia,ib,ic,id
      real, intent(in) :: kf_ks 
      real, dimension(3,Np), intent(in) :: r
      real, dimension(Np), intent(in) :: w
      real, dimension(2*Ngrid,Ngrid,Ngrid), intent(inout) :: dtl
      do 2 i=1,Np
        if (ia.eq.0 .and. ib.eq.0 .and. ic.eq.0 .and. id.eq.0) then !FFT delta
            we=w(i)
        elseif (ic.eq.0 .and. id.eq.0) then !FFT Qij
            rnorm=r(1,i)**2+r(2,i)**2+r(3,i)**2
            we=w(i)*r(ia,i)*r(ib,i)/rnorm
        else !FFT Qijkl
            rnorm=r(1,i)**2+r(2,i)**2+r(3,i)**2
            we=w(i)*r(ia,i)*r(ib,i)*r(ic,i)*r(id,i)/rnorm**2
        endif

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
       dtl(ixm2,iym2,izm2)   = dtl(ixm2,iym2,izm2)+ hxm2*hym2 *hzm2*we
       dtl(ixm1,iym2,izm2)   = dtl(ixm1,iym2,izm2)+ hxm1*hym2 *hzm2*we
       dtl(ixp1,iym2,izm2)   = dtl(ixp1,iym2,izm2)+ hxp1*hym2 *hzm2*we
       dtl(ixp2,iym2,izm2)   = dtl(ixp2,iym2,izm2)+ hxp2*hym2 *hzm2*we
       dtl(ixm2,iym1,izm2)   = dtl(ixm2,iym1,izm2)+ hxm2*hym1 *hzm2*we
       dtl(ixm1,iym1,izm2)   = dtl(ixm1,iym1,izm2)+ hxm1*hym1 *hzm2*we
       dtl(ixp1,iym1,izm2)   = dtl(ixp1,iym1,izm2)+ hxp1*hym1 *hzm2*we
       dtl(ixp2,iym1,izm2)   = dtl(ixp2,iym1,izm2)+ hxp2*hym1 *hzm2*we
       dtl(ixm2,iyp1,izm2)   = dtl(ixm2,iyp1,izm2)+ hxm2*hyp1 *hzm2*we
       dtl(ixm1,iyp1,izm2)   = dtl(ixm1,iyp1,izm2)+ hxm1*hyp1 *hzm2*we
       dtl(ixp1,iyp1,izm2)   = dtl(ixp1,iyp1,izm2)+ hxp1*hyp1 *hzm2*we
       dtl(ixp2,iyp1,izm2)   = dtl(ixp2,iyp1,izm2)+ hxp2*hyp1 *hzm2*we
       dtl(ixm2,iyp2,izm2)   = dtl(ixm2,iyp2,izm2)+ hxm2*hyp2 *hzm2*we
       dtl(ixm1,iyp2,izm2)   = dtl(ixm1,iyp2,izm2)+ hxm1*hyp2 *hzm2*we
       dtl(ixp1,iyp2,izm2)   = dtl(ixp1,iyp2,izm2)+ hxp1*hyp2 *hzm2*we
       dtl(ixp2,iyp2,izm2)   = dtl(ixp2,iyp2,izm2)+ hxp2*hyp2 *hzm2*we
       dtl(ixm2,iym2,izm1)   = dtl(ixm2,iym2,izm1)+ hxm2*hym2 *hzm1*we
       dtl(ixm1,iym2,izm1)   = dtl(ixm1,iym2,izm1)+ hxm1*hym2 *hzm1*we
       dtl(ixp1,iym2,izm1)   = dtl(ixp1,iym2,izm1)+ hxp1*hym2 *hzm1*we
       dtl(ixp2,iym2,izm1)   = dtl(ixp2,iym2,izm1)+ hxp2*hym2 *hzm1*we
       dtl(ixm2,iym1,izm1)   = dtl(ixm2,iym1,izm1)+ hxm2*hym1 *hzm1*we
       dtl(ixm1,iym1,izm1)   = dtl(ixm1,iym1,izm1)+ hxm1*hym1 *hzm1*we
       dtl(ixp1,iym1,izm1)   = dtl(ixp1,iym1,izm1)+ hxp1*hym1 *hzm1*we
       dtl(ixp2,iym1,izm1)   = dtl(ixp2,iym1,izm1)+ hxp2*hym1 *hzm1*we
       dtl(ixm2,iyp1,izm1)   = dtl(ixm2,iyp1,izm1)+ hxm2*hyp1 *hzm1*we
       dtl(ixm1,iyp1,izm1)   = dtl(ixm1,iyp1,izm1)+ hxm1*hyp1 *hzm1*we
       dtl(ixp1,iyp1,izm1)   = dtl(ixp1,iyp1,izm1)+ hxp1*hyp1 *hzm1*we
       dtl(ixp2,iyp1,izm1)   = dtl(ixp2,iyp1,izm1)+ hxp2*hyp1 *hzm1*we
       dtl(ixm2,iyp2,izm1)   = dtl(ixm2,iyp2,izm1)+ hxm2*hyp2 *hzm1*we
       dtl(ixm1,iyp2,izm1)   = dtl(ixm1,iyp2,izm1)+ hxm1*hyp2 *hzm1*we
       dtl(ixp1,iyp2,izm1)   = dtl(ixp1,iyp2,izm1)+ hxp1*hyp2 *hzm1*we
       dtl(ixp2,iyp2,izm1)   = dtl(ixp2,iyp2,izm1)+ hxp2*hyp2 *hzm1*we
       dtl(ixm2,iym2,izp1)   = dtl(ixm2,iym2,izp1)+ hxm2*hym2 *hzp1*we
       dtl(ixm1,iym2,izp1)   = dtl(ixm1,iym2,izp1)+ hxm1*hym2 *hzp1*we
       dtl(ixp1,iym2,izp1)   = dtl(ixp1,iym2,izp1)+ hxp1*hym2 *hzp1*we
       dtl(ixp2,iym2,izp1)   = dtl(ixp2,iym2,izp1)+ hxp2*hym2 *hzp1*we
       dtl(ixm2,iym1,izp1)   = dtl(ixm2,iym1,izp1)+ hxm2*hym1 *hzp1*we
       dtl(ixm1,iym1,izp1)   = dtl(ixm1,iym1,izp1)+ hxm1*hym1 *hzp1*we
       dtl(ixp1,iym1,izp1)   = dtl(ixp1,iym1,izp1)+ hxp1*hym1 *hzp1*we
       dtl(ixp2,iym1,izp1)   = dtl(ixp2,iym1,izp1)+ hxp2*hym1 *hzp1*we
       dtl(ixm2,iyp1,izp1)   = dtl(ixm2,iyp1,izp1)+ hxm2*hyp1 *hzp1*we
       dtl(ixm1,iyp1,izp1)   = dtl(ixm1,iyp1,izp1)+ hxm1*hyp1 *hzp1*we
       dtl(ixp1,iyp1,izp1)   = dtl(ixp1,iyp1,izp1)+ hxp1*hyp1 *hzp1*we
       dtl(ixp2,iyp1,izp1)   = dtl(ixp2,iyp1,izp1)+ hxp2*hyp1 *hzp1*we
       dtl(ixm2,iyp2,izp1)   = dtl(ixm2,iyp2,izp1)+ hxm2*hyp2 *hzp1*we
       dtl(ixm1,iyp2,izp1)   = dtl(ixm1,iyp2,izp1)+ hxm1*hyp2 *hzp1*we
       dtl(ixp1,iyp2,izp1)   = dtl(ixp1,iyp2,izp1)+ hxp1*hyp2 *hzp1*we
       dtl(ixp2,iyp2,izp1)   = dtl(ixp2,iyp2,izp1)+ hxp2*hyp2 *hzp1*we
       dtl(ixm2,iym2,izp2)   = dtl(ixm2,iym2,izp2)+ hxm2*hym2 *hzp2*we
       dtl(ixm1,iym2,izp2)   = dtl(ixm1,iym2,izp2)+ hxm1*hym2 *hzp2*we
       dtl(ixp1,iym2,izp2)   = dtl(ixp1,iym2,izp2)+ hxp1*hym2 *hzp2*we
       dtl(ixp2,iym2,izp2)   = dtl(ixp2,iym2,izp2)+ hxp2*hym2 *hzp2*we
       dtl(ixm2,iym1,izp2)   = dtl(ixm2,iym1,izp2)+ hxm2*hym1 *hzp2*we
       dtl(ixm1,iym1,izp2)   = dtl(ixm1,iym1,izp2)+ hxm1*hym1 *hzp2*we
       dtl(ixp1,iym1,izp2)   = dtl(ixp1,iym1,izp2)+ hxp1*hym1 *hzp2*we
       dtl(ixp2,iym1,izp2)   = dtl(ixp2,iym1,izp2)+ hxp2*hym1 *hzp2*we
       dtl(ixm2,iyp1,izp2)   = dtl(ixm2,iyp1,izp2)+ hxm2*hyp1 *hzp2*we
       dtl(ixm1,iyp1,izp2)   = dtl(ixm1,iyp1,izp2)+ hxm1*hyp1 *hzp2*we
       dtl(ixp1,iyp1,izp2)   = dtl(ixp1,iyp1,izp2)+ hxp1*hyp1 *hzp2*we
       dtl(ixp2,iyp1,izp2)   = dtl(ixp2,iyp1,izp2)+ hxp2*hyp1 *hzp2*we
       dtl(ixm2,iyp2,izp2)   = dtl(ixm2,iyp2,izp2)+ hxm2*hyp2 *hzp2*we
       dtl(ixm1,iyp2,izp2)   = dtl(ixm1,iyp2,izp2)+ hxm1*hyp2 *hzp2*we
       dtl(ixp1,iyp2,izp2)   = dtl(ixp1,iyp2,izp2)+ hxp1*hyp2 *hzp2*we
       dtl(ixp2,iyp2,izp2)   = dtl(ixp2,iyp2,izp2)+ hxp2*hyp2 *hzp2*we
c
       dtl(nxm2,nym2,nzm2)   = dtl(nxm2,nym2,nzm2)+ gxm2*gym2 *gzm2*we
       dtl(nxm1,nym2,nzm2)   = dtl(nxm1,nym2,nzm2)+ gxm1*gym2 *gzm2*we
       dtl(nxp1,nym2,nzm2)   = dtl(nxp1,nym2,nzm2)+ gxp1*gym2 *gzm2*we
       dtl(nxp2,nym2,nzm2)   = dtl(nxp2,nym2,nzm2)+ gxp2*gym2 *gzm2*we
       dtl(nxm2,nym1,nzm2)   = dtl(nxm2,nym1,nzm2)+ gxm2*gym1 *gzm2*we
       dtl(nxm1,nym1,nzm2)   = dtl(nxm1,nym1,nzm2)+ gxm1*gym1 *gzm2*we
       dtl(nxp1,nym1,nzm2)   = dtl(nxp1,nym1,nzm2)+ gxp1*gym1 *gzm2*we
       dtl(nxp2,nym1,nzm2)   = dtl(nxp2,nym1,nzm2)+ gxp2*gym1 *gzm2*we
       dtl(nxm2,nyp1,nzm2)   = dtl(nxm2,nyp1,nzm2)+ gxm2*gyp1 *gzm2*we
       dtl(nxm1,nyp1,nzm2)   = dtl(nxm1,nyp1,nzm2)+ gxm1*gyp1 *gzm2*we
       dtl(nxp1,nyp1,nzm2)   = dtl(nxp1,nyp1,nzm2)+ gxp1*gyp1 *gzm2*we
       dtl(nxp2,nyp1,nzm2)   = dtl(nxp2,nyp1,nzm2)+ gxp2*gyp1 *gzm2*we
       dtl(nxm2,nyp2,nzm2)   = dtl(nxm2,nyp2,nzm2)+ gxm2*gyp2 *gzm2*we
       dtl(nxm1,nyp2,nzm2)   = dtl(nxm1,nyp2,nzm2)+ gxm1*gyp2 *gzm2*we
       dtl(nxp1,nyp2,nzm2)   = dtl(nxp1,nyp2,nzm2)+ gxp1*gyp2 *gzm2*we
       dtl(nxp2,nyp2,nzm2)   = dtl(nxp2,nyp2,nzm2)+ gxp2*gyp2 *gzm2*we
       dtl(nxm2,nym2,nzm1)   = dtl(nxm2,nym2,nzm1)+ gxm2*gym2 *gzm1*we
       dtl(nxm1,nym2,nzm1)   = dtl(nxm1,nym2,nzm1)+ gxm1*gym2 *gzm1*we
       dtl(nxp1,nym2,nzm1)   = dtl(nxp1,nym2,nzm1)+ gxp1*gym2 *gzm1*we
       dtl(nxp2,nym2,nzm1)   = dtl(nxp2,nym2,nzm1)+ gxp2*gym2 *gzm1*we
       dtl(nxm2,nym1,nzm1)   = dtl(nxm2,nym1,nzm1)+ gxm2*gym1 *gzm1*we
       dtl(nxm1,nym1,nzm1)   = dtl(nxm1,nym1,nzm1)+ gxm1*gym1 *gzm1*we
       dtl(nxp1,nym1,nzm1)   = dtl(nxp1,nym1,nzm1)+ gxp1*gym1 *gzm1*we
       dtl(nxp2,nym1,nzm1)   = dtl(nxp2,nym1,nzm1)+ gxp2*gym1 *gzm1*we
       dtl(nxm2,nyp1,nzm1)   = dtl(nxm2,nyp1,nzm1)+ gxm2*gyp1 *gzm1*we
       dtl(nxm1,nyp1,nzm1)   = dtl(nxm1,nyp1,nzm1)+ gxm1*gyp1 *gzm1*we
       dtl(nxp1,nyp1,nzm1)   = dtl(nxp1,nyp1,nzm1)+ gxp1*gyp1 *gzm1*we
       dtl(nxp2,nyp1,nzm1)   = dtl(nxp2,nyp1,nzm1)+ gxp2*gyp1 *gzm1*we
       dtl(nxm2,nyp2,nzm1)   = dtl(nxm2,nyp2,nzm1)+ gxm2*gyp2 *gzm1*we
       dtl(nxm1,nyp2,nzm1)   = dtl(nxm1,nyp2,nzm1)+ gxm1*gyp2 *gzm1*we
       dtl(nxp1,nyp2,nzm1)   = dtl(nxp1,nyp2,nzm1)+ gxp1*gyp2 *gzm1*we
       dtl(nxp2,nyp2,nzm1)   = dtl(nxp2,nyp2,nzm1)+ gxp2*gyp2 *gzm1*we
       dtl(nxm2,nym2,nzp1)   = dtl(nxm2,nym2,nzp1)+ gxm2*gym2 *gzp1*we
       dtl(nxm1,nym2,nzp1)   = dtl(nxm1,nym2,nzp1)+ gxm1*gym2 *gzp1*we
       dtl(nxp1,nym2,nzp1)   = dtl(nxp1,nym2,nzp1)+ gxp1*gym2 *gzp1*we
       dtl(nxp2,nym2,nzp1)   = dtl(nxp2,nym2,nzp1)+ gxp2*gym2 *gzp1*we
       dtl(nxm2,nym1,nzp1)   = dtl(nxm2,nym1,nzp1)+ gxm2*gym1 *gzp1*we
       dtl(nxm1,nym1,nzp1)   = dtl(nxm1,nym1,nzp1)+ gxm1*gym1 *gzp1*we
       dtl(nxp1,nym1,nzp1)   = dtl(nxp1,nym1,nzp1)+ gxp1*gym1 *gzp1*we
       dtl(nxp2,nym1,nzp1)   = dtl(nxp2,nym1,nzp1)+ gxp2*gym1 *gzp1*we
       dtl(nxm2,nyp1,nzp1)   = dtl(nxm2,nyp1,nzp1)+ gxm2*gyp1 *gzp1*we
       dtl(nxm1,nyp1,nzp1)   = dtl(nxm1,nyp1,nzp1)+ gxm1*gyp1 *gzp1*we
       dtl(nxp1,nyp1,nzp1)   = dtl(nxp1,nyp1,nzp1)+ gxp1*gyp1 *gzp1*we
       dtl(nxp2,nyp1,nzp1)   = dtl(nxp2,nyp1,nzp1)+ gxp2*gyp1 *gzp1*we
       dtl(nxm2,nyp2,nzp1)   = dtl(nxm2,nyp2,nzp1)+ gxm2*gyp2 *gzp1*we
       dtl(nxm1,nyp2,nzp1)   = dtl(nxm1,nyp2,nzp1)+ gxm1*gyp2 *gzp1*we
       dtl(nxp1,nyp2,nzp1)   = dtl(nxp1,nyp2,nzp1)+ gxp1*gyp2 *gzp1*we
       dtl(nxp2,nyp2,nzp1)   = dtl(nxp2,nyp2,nzp1)+ gxp2*gyp2 *gzp1*we
       dtl(nxm2,nym2,nzp2)   = dtl(nxm2,nym2,nzp2)+ gxm2*gym2 *gzp2*we
       dtl(nxm1,nym2,nzp2)   = dtl(nxm1,nym2,nzp2)+ gxm1*gym2 *gzp2*we
       dtl(nxp1,nym2,nzp2)   = dtl(nxp1,nym2,nzp2)+ gxp1*gym2 *gzp2*we
       dtl(nxp2,nym2,nzp2)   = dtl(nxp2,nym2,nzp2)+ gxp2*gym2 *gzp2*we
       dtl(nxm2,nym1,nzp2)   = dtl(nxm2,nym1,nzp2)+ gxm2*gym1 *gzp2*we
       dtl(nxm1,nym1,nzp2)   = dtl(nxm1,nym1,nzp2)+ gxm1*gym1 *gzp2*we
       dtl(nxp1,nym1,nzp2)   = dtl(nxp1,nym1,nzp2)+ gxp1*gym1 *gzp2*we
       dtl(nxp2,nym1,nzp2)   = dtl(nxp2,nym1,nzp2)+ gxp2*gym1 *gzp2*we
       dtl(nxm2,nyp1,nzp2)   = dtl(nxm2,nyp1,nzp2)+ gxm2*gyp1 *gzp2*we
       dtl(nxm1,nyp1,nzp2)   = dtl(nxm1,nyp1,nzp2)+ gxm1*gyp1 *gzp2*we
       dtl(nxp1,nyp1,nzp2)   = dtl(nxp1,nyp1,nzp2)+ gxp1*gyp1 *gzp2*we
       dtl(nxp2,nyp1,nzp2)   = dtl(nxp2,nyp1,nzp2)+ gxp2*gyp1 *gzp2*we
       dtl(nxm2,nyp2,nzp2)   = dtl(nxm2,nyp2,nzp2)+ gxm2*gyp2 *gzp2*we
       dtl(nxm1,nyp2,nzp2)   = dtl(nxm1,nyp2,nzp2)+ gxm1*gyp2 *gzp2*we
       dtl(nxp1,nyp2,nzp2)   = dtl(nxp1,nyp2,nzp2)+ gxp1*gyp2 *gzp2*we
       dtl(nxp2,nyp2,nzp2)   = dtl(nxp2,nyp2,nzp2)+ gxp2*gyp2 *gzp2*we
2     continue
c
      return
      end
cc*******************************************************************
      subroutine FiveDelta2g_1(dcgxx,dcgyy,dcgzz,Ngrid)
cc*******************************************************************
      integer ix,ikx,iy,iky,iz,ikz
      real rk,amu
      integer, intent(in) :: Ngrid
      complex, intent(in) :: dcgyy(Ngrid/2+1,Ngrid,Ngrid)
      complex, intent(in) :: dcgzz(Ngrid/2+1,Ngrid,Ngrid)
      complex, intent(inout) :: dcgxx(Ngrid/2+1,Ngrid,Ngrid)
      
      do 104 iz=1,Ngrid !build quadrupole
         ikz=mod(iz+Ngrid/2-2,Ngrid)-Ngrid/2+1
         do 104 iy=1,Ngrid
            iky=mod(iy+Ngrid/2-2,Ngrid)-Ngrid/2+1
            do 104 ix=1,Ngrid/2+1
               ikx=mod(ix+Ngrid/2-2,Ngrid)-Ngrid/2+1
               rk=sqrt(float(ikx**2+iky**2+ikz**2))
               if(rk.gt.0.)then
                  kxh=float(ikx)/rk
                  kyh=float(iky)/rk
                  kzh=float(ikz)/rk
                  dcgxx(ix,iy,iz)=7.5*(dcgxx(ix,iy,iz)*kxh**2 
     &                  +dcgyy(ix,iy,iz)*kyh**2
     &                  +dcgzz(ix,iy,iz)*kzh**2)
               end if
 104  continue
      return 
      end 
cc*******************************************************************
      subroutine FiveDelta2g_2(dcg, dcgxx,dcgxy,dcgyz,dcgzx,Ngrid)
cc*******************************************************************
      integer ix,ikx,iy,iky,iz,ikz
      real rk,amu
      integer, intent(in) :: Ngrid
      complex, intent(in) :: dcg(Ngrid/2+1,Ngrid,Ngrid)
      complex, intent(in) :: dcgxy(Ngrid/2+1,Ngrid,Ngrid)
      complex, intent(in) :: dcgyz(Ngrid/2+1,Ngrid,Ngrid)
      complex, intent(in) :: dcgzx(Ngrid/2+1,Ngrid,Ngrid)
      complex, intent(inout) :: dcgxx(Ngrid/2+1,Ngrid,Ngrid)
      
      do 104 iz=1,Ngrid !build quadrupole
         ikz=mod(iz+Ngrid/2-2,Ngrid)-Ngrid/2+1
         do 104 iy=1,Ngrid
            iky=mod(iy+Ngrid/2-2,Ngrid)-Ngrid/2+1
            do 104 ix=1,Ngrid/2+1
               ikx=mod(ix+Ngrid/2-2,Ngrid)-Ngrid/2+1
               rk=sqrt(float(ikx**2+iky**2+ikz**2))
               if(rk.gt.0.)then
                  kxh=float(ikx)/rk
                  kyh=float(iky)/rk
                  kzh=float(ikz)/rk
                  dcgxx(ix,iy,iz)=dcgxx(ix,iy,iz) + 7.5*( 
     &                   2.*dcgxy(ix,iy,iz)*kxh*kyh
     &                  +2.*dcgyz(ix,iy,iz)*kyh*kzh
     &                  +2.*dcgzx(ix,iy,iz)*kzh*kxh)
     &                  -2.5*dcg(ix,iy,iz)   
               end if
 104  continue
      return 
      end 
cc*******************************************************************
      subroutine build_quad(dclr1,dclr2,irsd,Ngrid)
cc*******************************************************************
      integer ix,ikx,iy,iky,iz,ikz
      real rk,amu
      integer, intent(in) :: irsd,Ngrid
      complex, intent(in) :: dclr1(Ngrid/2+1,Ngrid,Ngrid)
      complex, intent(inout) :: dclr2(Ngrid/2+1,Ngrid,Ngrid)
      
      do 100 iz=1,Ngrid !build quadrupole
         ikz=mod(iz+Ngrid/2-2,Ngrid)-Ngrid/2+1
         do 100 iy=1,Ngrid
            iky=mod(iy+Ngrid/2-2,Ngrid)-Ngrid/2+1
            do 100 ix=1,Ngrid/2+1
               ikx=mod(ix+Ngrid/2-2,Ngrid)-Ngrid/2+1
               rk=sqrt(float(ikx**2+iky**2+ikz**2))
               if(rk.gt.0.)then
                  if (irsd.eq.3) then 
                     amu=float(ikz)/rk
                  elseif (irsd.eq.2) then
                     amu=float(iky)/rk
                  elseif (irsd.eq.1) then
                     amu=float(ikx)/rk !unit vectors
                  else
                     stop
                  endif   
                  dclr2(ix,iy,iz)=(7.5*amu**2-2.5)*dclr1(ix,iy,iz) !weight by 5*Leg2
               end if
 100  continue
      return 
      end 
cc*******************************************************************
      subroutine fcomb_periodic(dcl,N,Ngrid)
cc*******************************************************************
      parameter(tpi=6.283185307d0)
      real*8 tpiL,piL
      complex*16 recx,recy,recz,xrec,yrec,zrec
      complex c1,ci,c000,c001,c010,c011,cma,cmb,cmc,cmd
      real, intent(in) :: N
      integer, intent(in) :: Ngrid
      complex, intent(inout) :: dcl(Ngrid,Ngrid,Ngrid)

      cf=1./(6.**3*4.*N)
c      cf=1./(6.**3*4.) !*float(N)) not needed for FKP

      Lnyq=Ngrid/2+1
      tpiL=tpi/float(Ngrid)
      piL=-tpiL/2.
      recx=cmplx(dcos(piL),dsin(piL))
      recy=cmplx(dcos(piL),dsin(piL))
      recz=cmplx(dcos(piL),dsin(piL))
      
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

