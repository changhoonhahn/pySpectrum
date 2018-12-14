ccc*******************************************************************
c      subroutine bk_periodic(map1,map2,nside,step,ncut,nmax)
ccc*******************************************************************
c      integer nside,nsideD,Dim,nn(3),nmax,istep
c      real, intent(in) :: map1w(:),map2w(:)
c      parameter(Dim=3) 
c      integer*8 planb,planf
c      integer i, j, l, m, n, nk(0:2*nside),nbk(0:2*nside+1),iflag,irsd
c      integer iseed, indx(nsideD), id, jd, ld, nmodes, ndim,k,ix,iy,iz
cc      real mapk(nsideD,nmax),m1(2,nsideD), map1(nsideD), map2(nsideD)  
c      real pow(nmax),I10,I12,I22,I13,I23,I33,P0,alpha
c      real pow2(nmax), I12d,I13d,alphaSTD, poww(nmax)
c      real, allocatable :: mapk(:,:),m1(:,:),map1(:),map2(:)
c      real, allocatable :: m1xx(:,:),map1xx(:),map2xx(:)
c      real, allocatable :: m1w(:,:),map1w(:),map2w(:)
cc      real normk(nsideD,nmax), norm1(2,nsideD)
c      real di, dj, dl, eightpi2, bi, step, bi2, biw
c      real dist, bisp(nmax,nmax,nmax), q(nmax,nmax,nmax)
c      real bisp2(nmax,nmax,nmax), q2(nmax,nmax,nmax)
c      real bispw(nmax,nmax,nmax) 
c      real*8 coun(nmax,nmax,nmax), avg, sum, sum2, avgw
c      complex, allocatable :: dclr1(:,:,:),dclr2(:,:,:)
c      complex, allocatable :: dclr3(:,:,:),dclr4(:,:,:)
c      complex, allocatable :: dclr5(:,:,:),dclr6(:,:,:)
c      integer nobj,ip,aexp,p,dpnp1,dpn,ibuf,lm,mcode,Ncut,ng,Ncuts
c      common/discrete/I10,I12,I22,I13,I23,I33,P0,alpha
c      common/discrete2/I12d,I13d,alphaSTD
c      include '/usr/local/include/fftw3.f'
c
c      ncuts=ncut/int(step)
c      nsideD=nside**Dim
c      eightpi2 = 78.95683521
c     
c      allocate(map1(nside**Dim),map2(nside**Dim))
c
cc      write(*,*) 'Nbody/periodic (1) or data/cut-sky (2)?'
cc      read(*,*)iflag
c      call getarg(1,iflagstr)
c      read(iflagstr,*)iflag 
c
cc         write(*,*) 'Fourier file :'
cc         read(*,'(a)') filecoef
cc         write(*,*) 'Bispectrum file :'
cc         read(*,'(a)') filebisp
c      call getarg(2,filecoef)
c      call getarg(3,filebisp)
c      call getarg(4,irsdstr) !redshift-space direction (1:x,2:y,3:z)
c      read(irsdstr,*)irsd 
c
c      allocate(dclr1(nside/2+1,nside,nside))
c      allocate(dclr2(nside/2+1,nside,nside))
c      write(*,*)'memory allocated'
cc      call inputNB(nside,Dim,filecoef,map1,map2,dclr1)
c      call inputNB(nside,Dim,filecoef,map1,map2,dclr1,dclr2,
c     $                map1xx,map2xx,irsd)
c      deallocate(dclr1,dclr2)
c      write(*,*)'input done and memory deallocated'
c
c      do l=1,nmax
c         do j=1,nmax
c            do i=1,nmax
c               bisp(i,j,l) = 0.
c               bisp2(i,j,l) = 0.
c            enddo
c         enddo
c      enddo
c      
c      do i=0,2*nside
c         nk(i)=0
c      enddo
c
c      write(*,*) 'Find modes of amplitude |k|'
c
c      do i = 0, nside -1
c         do j = 0, nside  -1
c            do l = 0, nside  -1
c               di = float(min(i,nside-i))
c               dj = float(min(j,nside-j))
c               dl = float(min(l,nside-l))
c               dist = sqrt(di*di + dj*dj +dl*dl)
c               nk(int(dist/step+0.5)) = nk(int(dist/step+0.5)) + 1 
c            enddo 
c         enddo 
c      enddo 
c      
c      nbk(0) = 0 
c      do i = 0, 2*nside
c         nbk(i+1) = nbk(i) + nk(i)
c         nk(i) = 0 
c      enddo 
c      
c      write(*,*) 'Save coordinates'
c  
c      m = 0
c      do i = 0, nside -1 
c         do j = 0, nside -1
c            do l = 0, nside -1
c               di = float(min(i,nside-i))
c               dj = float(min(j,nside-j))
c               dl = float(min(l,nside-l))
c               dist = sqrt(di*di + dj*dj +dl*dl)
c               nk(int(dist/step+0.5)) = nk(int(dist/step+0.5)) + 1  
c               n = nbk(int(dist/step+0.5)) + nk(int(dist/step+0.5))
c               m = m+ 1
c               indx(n) = m 
c            enddo 
c         enddo 
c      enddo 
c
c      write(*,*) 'Calculate k maps'
c
c      ndim=3
c
c      allocate(m1(2,nsideD),mapk(nsideD,nmax),m1w(2,nsideD))
c      do i = ncuts, nmax
c         do n = 1, nsideD 
c            m1(1,n) = 0.
c            m1(2,n) = 0.
c         enddo 
c         nmodes = 0
c         do n = nbk(i)+1, nbk(i+1)
c            m1(1,indx(n)) = map1(indx(n))
c            m1(2,indx(n)) = map2(indx(n))
c            nmodes =nmodes +1 
c         enddo 
c         call fftwnd_f77_one(planf,m1,0)
c         avg = 0.d0 
c         do n = 1, nsideD
c            avg = avg + dble(m1(1,n))*dble(m1(1,n))
c            mapk(n,i) = m1(1,n)
c         enddo
c         pow(i)=real(avg)/float(nsided)/float(nmodes)
c      enddo
c      
c      deallocate(m1,map1)
c
c      write(*,*) 'Read counts'
c
c      open(unit=2,status='old',form='unformatted',file=filecounts)
c      read(2) coun
c      close(2)
c
c      write(*,*) 'counts were read'
c      allocate(m1xx(2,nsideD))
c
c      write(*,*) 'Compute bisp monopole and quadrupole!'
c      
c      
c      do l = ncuts, nmax !keep ffts in outer loop
c         do n = 1, nsideD 
c            m1xx(1,n) = 0.        
c            m1xx(2,n) = 0.        
c         enddo
c         nmodes = 0
c         do n = nbk(l)+1, nbk(l+1)
c            m1xx(1,indx(n)) = map1xx(indx(n))
c            m1xx(2,indx(n)) = map2xx(indx(n))
c            nmodes = nmodes + 1 
c         enddo 
c         call fftwnd_f77_one(planf,m1xx,0)
c         avg = 0.d0 
c         do n = 1, nsideD
c            avg = avg + dble(m1xx(1,n))*dble(mapk(n,l))
c         enddo
c         pow2(l)=real(avg)/float(nsided)/float(nmodes) !quadrupole power
c
c         do j = ncuts, l
c            do i = max(ncuts,l-j), j
c               sum = 0.d0 
c               sum2 = 0.d0 
c               do n = 1, nsideD
c                  sum = sum 
c     $            + dble(mapk(n,i))*dble(mapk(n,j))*dble(mapk(n,l))
c                  sum2 = sum2 
c     $            + dble(mapk(n,i))*dble(mapk(n,j))*dble(m1xx(1,n))
c               enddo
c               bi=real(sum/coun(i,j,l))
c               bi2=real(sum2/coun(i,j,l))
c               bisp(i,j,l)=bi
c               q(i,j,l)=bi/(pow(i)*pow(j)+pow(j)*pow(l)
c     $                 +pow(l)*pow(i))
c               bisp2(i,j,l)=bi2
c               q2(i,j,l)=bi2/(pow(i)*pow(j)+pow(j)*pow(l)
c     $                 +pow(l)*pow(i))
c             enddo
c         enddo 
c      enddo          
c      
c      open(unit=7,file=filebisp,status='unknown',form='formatted')
c      write(*,*) 'output'
c      do l = ncuts, nmax
c         do j = ncuts, l
c            do i = ncuts,j
c               fac=1.
c               if(coun(i,j,l).ne.0.d0) then 
c                  if(j.eq.l .and. i.eq.j) fac=6.
c                  if(i.eq.j .and. j.ne.l) fac=2.
c                  if(i.eq.l .and. l.ne.j) fac=2.
c                  if(j.eq.l .and. l.ne.i) fac=2.
c                  coun(i,j,l)=coun(i,j,l)/dble(fac*float(nsided))
c                  if (iflag.eq.1) then 
c                  write(7,1000) int(step)*l,int(step)*j,int(step)*i,
c     $                 pow(l),pow(j),pow(i),bisp(i,j,l),q(i,j,l),
c     $                 pow2(l),pow2(j),pow2(i),bisp2(i,j,l),q2(i,j,l),
c     $                 real(coun(i,j,l))
c                  else
c                  write(7,1000) int(step)*l,int(step)*j,int(step)*i,
c     $                 pow(l),pow(j),pow(i),bisp(i,j,l),q(i,j,l),
c     $                 pow2(l),pow2(j),pow2(i),bisp2(i,j,l),q2(i,j,l),
c     $                 real(coun(i,j,l)),bispw(i,j,l)                  
c                  endif
c               endif
c            enddo
c         enddo
c      enddo 
c      close(7)
c      return    
c      end 
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
      include '/usr/local/include/fftw3.f'

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
      parameter(tpi=6.283185307d0)
      real*8 tpiL,piL
      complex*16 recx,recy,recz,xrec,yrec,zrec
      complex c1,ci,c000,c001,c010,c011,cma,cmb,cmc,cmd
      integer, intent(in) :: N,Ngrid
      complex, intent(inout) :: dcl(Ngrid,Ngrid,Ngrid)

      cf=1./(6.**3*4.*N)

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

