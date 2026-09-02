! TSVC_2 vtvtv: a(i) = a(i) * b(i) * c(i), elementwise over LEN_1D.
! C-ABI per hpcagent_bench c-abi-v2: pointers (a,b,c) name-sorted, then the
! int64 size symbol LEN_1D, then the trailing reserved workspace pair.
! Explicit (a*b)*c via temps keeps results bitwise-identical to the numpy
! reference ((a*b)*c); 16-wide unroll gives the vectorizer independent wide
! chains; dynamic 16K-element chunks balance the 24 pinned threads and spread
! their streams (harness large arrays are 4K-aligned, so 64 B vector groups
! starting at i=1,17,... are aligned).
subroutine tsvc_2_vtvtv_fp64(a, b, c, len_1d, workspace, workspace_size) &
     bind(C, name="tsvc_2_vtvtv_fp64")
  use, intrinsic :: iso_c_binding, only: c_int64_t, c_int8_t, c_double
  implicit none
  real(c_double), intent(inout)      :: a(*)
  real(c_double), intent(in)         :: b(*)
  real(c_double), intent(in)         :: c(*)
  integer(c_int64_t), value          :: len_1d
  integer(c_int8_t),  intent(in)     :: workspace(*)
  integer(c_int64_t), value          :: workspace_size
  integer(c_int64_t) :: n, main_end, i
  real(c_double)     :: t, t1, t2, t3, t4, t5, t6, t7, t8, t9
  real(c_double)     :: t10, t11, t12, t13, t14, t15, t16
  n = len_1d
  if (n <= 65536_8) then
     do i = 1_8, n
        t = a(i) * b(i)
        a(i) = t * c(i)
     end do
  else
     main_end = (n / 16_8) * 16_8
     !$omp parallel do schedule(dynamic, 16384)
     do i = 1_8, main_end, 16
        t1  = a(i)*b(i);      a(i)      = t1*c(i)
        t2  = a(i+1)*b(i+1);  a(i+1)    = t2*c(i+1)
        t3  = a(i+2)*b(i+2);  a(i+2)    = t3*c(i+2)
        t4  = a(i+3)*b(i+3);  a(i+3)    = t4*c(i+3)
        t5  = a(i+4)*b(i+4);  a(i+4)    = t5*c(i+4)
        t6  = a(i+5)*b(i+5);  a(i+5)    = t6*c(i+5)
        t7  = a(i+6)*b(i+6);  a(i+6)    = t7*c(i+6)
        t8  = a(i+7)*b(i+7);  a(i+7)    = t8*c(i+7)
        t9  = a(i+8)*b(i+8);  a(i+8)    = t9*c(i+8)
        t10 = a(i+9)*b(i+9);  a(i+9)    = t10*c(i+9)
        t11 = a(i+10)*b(i+10); a(i+10)  = t11*c(i+10)
        t12 = a(i+11)*b(i+11); a(i+11)  = t12*c(i+11)
        t13 = a(i+12)*b(i+12); a(i+12)  = t13*c(i+12)
        t14 = a(i+13)*b(i+13); a(i+13)  = t14*c(i+13)
        t15 = a(i+14)*b(i+14); a(i+14)  = t15*c(i+14)
        t16 = a(i+15)*b(i+15); a(i+15)  = t16*c(i+15)
     end do
     !$omp end parallel do
     do i = main_end + 1, n
        t = a(i) * b(i)
        a(i) = t * c(i)
     end do
  end if
end subroutine tsvc_2_vtvtv_fp64
